import torch
from .utils import split_tensors,repeat_tensors,penalty_builder
import torch.nn.functional as F

class BeamSearch:
    def __init__(self, config):
        super().__init__()
        #self.model = pl_module
        self.beam_size = config['beam_size']
        self.group_size = config['group_size']
        self.sample_n = config['sample_n']
        self.sample_method = config['sample_method']
        self.temperature = config['temperature']
        self.output_logsoftmax = config['output_logsoftmax']
        self.decoding_constraint = config['decoding_constraint']
        self.block_trigrams = config['block_trigrams']
        self.diversity_lambda = config['diversity_lambda']
        self.length_penalty = config['length_penalty']
        self.suppress_UNK = config['suppress_UNK']
        self.max_seq_length = config['max_seq_length']

        self.vocab_size = config['vocab_size']
        self.eos_idx = config['eos_idx']
        self.pad_idx = config['pad_idx']
        self.bos_idx = config['bos_idx']

    def set_tokenidx(self,tokenizer):
        self.pad_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.bos_idx = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.eos_idx = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        self.sep_idx = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        self.vocab_size = tokenizer.vocab_size


    def init_hidden(self, bsz):
        return []

    def sample(self, model, patch_feats, mask=None):
        sample_method = self.sample_method
        beam_size = self.beam_size
        temperature = self.temperature
        sample_n = int(self.sample_n)
        group_size = self.group_size
        output_logsoftmax = self.output_logsoftmax
        decoding_constraint = self.decoding_constraint
        block_trigrams = self.block_trigrams
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            return self._sample_beam(model, patch_feats, mask)
        # if group_size > 1:
        #     return self._diverse_sample(model,img_feats)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size * sample_n)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        if sample_n > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = repeat_tensors(sample_n,
                                                                                      [p_fc_feats, p_att_feats,
                                                                                       pp_att_feats, p_att_masks]
                                                                                      )

        trigrams = []  # will be a list of batch_size dictionaries

        seq = fc_feats.new_full((batch_size * sample_n, self.max_seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.max_seq_length, self.vocab_size)
        for t in range(self.max_seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_full([batch_size * sample_n], self.bos_idx, dtype=torch.long)

            logprobs, state = model.core(it, p_att_feats, pp_att_feats, p_att_masks, state,
                                                      output_logsoftmax=output_logsoftmax)

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # Mess with trigrams
            # Copy from https://github.com/lukemelas/image-paragraph-captioning
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:, t - 3:t - 1]
                for i in range(batch_size):  # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current = seq[i][t - 1]
                    if t == 3:  # initialize
                        trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]:  # add to list
                            trigrams[i][prev_two].append(current)
                        else:  # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:, t - 2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()  # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i, j] += 1
                # Apply mask to log probs
                # logprobs = logprobs - (mask * 1e9)
                alpha = 2.0  # = 4
                logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == self.max_seq_length:  # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it != self.eos_idx
            else:
                it[~unfinished] = self.pad_idx  # This allows eos_idx not being overwritten to 0
                logprobs = logprobs * unfinished.unsqueeze(1).float()
                unfinished = unfinished * (it != self.eos_idx)
            seq[:, t] = it
            seqLogprobs[:, t] = logprobs
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs



    def _sample_beam(self, model, patch_feats, mask=None):
        beam_size = self.beam_size
        group_size = self.group_size
        sample_n = self.sample_n
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = patch_feats.size(0)

        #p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self.model.prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = patch_feats.new_full((batch_size * sample_n, self.max_seq_length), self.bos_idx, dtype=torch.long)
        # why plus 1 here?
        seqLogprobs = patch_feats.new_zeros(batch_size * sample_n, self.max_seq_length, self.vocab_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]

        state = self.init_hidden(batch_size)

        # first step, feed bos
        it = patch_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
        logprobs, state = model.core(it, patch_feats, mask, state)

        #img_feats = repeat_tensors(beam_size,[img_feats]) # use this for multiple values

        patch_feats, mask = repeat_tensors(beam_size, [patch_feats,mask])
        self.done_beams = self.beam_search(model, state, logprobs, patch_feats=patch_feats, mask=mask)
        # attns = []
        for k in range(batch_size):
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq_len = self.done_beams[k][_n]['seq'].shape[0]
                    seq[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['seq']
                    # if self.vis:
                    #     attns.append(self.done_beams[k][_n]['attn'])
                    seqLogprobs[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['logps']
            else:
                seq_len = self.done_beams[k][0]['seq'].shape[0]
                seq[k, :seq_len] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
                seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
                # if self.vis:
                #     attns.append(self.done_beams[k][0]['attn'])
        # return the samples and their log likelihoods
        # return seq, seqLogprobs,attns
        # return {'preds':seq.detach().cpu(), 'preds_log':seqLogprobs.detach().cpu()}
        return {'preds': seq.detach().cpu()}



    def beam_search(self, model, init_state, init_logprobs,patch_feats=None,mask=None):

        # function computes the similarity score to be augmented
        def add_diversity(beam_seq_table, logprobs, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobs = logprobs.clone()
            batch_size = beam_seq_table[0].shape[0]

            if divm > 0:
                change = logprobs.new_zeros(batch_size, logprobs.shape[-1])
                for prev_choice in range(divm):
                    prev_decisions = beam_seq_table[prev_choice][:, :, local_time]  # Nxb
                    for prev_labels in range(bdash):
                        change.scatter_add_(1, prev_decisions[:, prev_labels].unsqueeze(-1),
                                            change.new_ones(batch_size, 1))

                if local_time == 0:
                    logprobs = logprobs - change * diversity_lambda
                else:
                    logprobs = logprobs - self.repeat_tensor(bdash, change) * diversity_lambda

            return logprobs, unaug_logprobs

        # does one step of classical beam search

        def beam_step(logprobs, unaug_logprobs, beam_size, t, beam_seq, beam_seq_logprobs,
                      beam_logprobs_sum, state):
            # INPUTS:
            # logprobs: probabilities augmented after diversity N*bxV
            # beam_size: obvious
            # t        : time instant
            # beam_seq : tensor contanining the beams
            # beam_seq_logprobs: tensor contanining the beam logprobs
            # beam_logprobs_sum: tensor contanining joint logprobs
            # OUPUTS:
            # beam_seq : tensor containing the word indices of the decoded captions Nxbxl
            # beam_seq_logprobs : log-probability of each decision made, NxbxlxV
            # beam_logprobs_sum : joint log-probability of each beam Nxb

            batch_size = beam_logprobs_sum.shape[0]
            vocab_size = logprobs.shape[-1]
            logprobs = logprobs.reshape(batch_size, -1, vocab_size)  # NxbxV
            if t == 0:
                assert logprobs.shape[1] == 1
                beam_logprobs_sum = beam_logprobs_sum[:, :1]
            candidate_logprobs = beam_logprobs_sum.unsqueeze(-1) + logprobs  # beam_logprobs_sum Nxb logprobs is NxbxV
            # ys, ix = torch.sort(candidate_logprobs.reshape(candidate_logprobs.shape[0], -1), -1, True)
            # ys, ix = ys[:, :beam_size], ix[:, :beam_size]
            ys, ix = torch.topk(candidate_logprobs.reshape(candidate_logprobs.shape[0], -1), beam_size, -1, True)
            # print(ys, ix)
            # exit()
            # beam_ix = ix // vocab_size  # Nxb which beam
            # pytorch version difference
            beam_ix = torch.div(ix, vocab_size, rounding_mode='floor')
            selected_ix = ix % vocab_size  # Nxb # which world
            state_ix = (beam_ix + torch.arange(batch_size).type_as(beam_ix).unsqueeze(-1) * logprobs.shape[1]).reshape(
                -1)  # N*b which in Nxb beams

            if t > 0:
                # gather according to beam_ix
                assert (beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq)) ==
                        beam_seq.reshape(-1, beam_seq.shape[-1])[state_ix].view_as(beam_seq)).all()
                beam_seq = beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq))

                beam_seq_logprobs = beam_seq_logprobs.gather(1, beam_ix.unsqueeze(-1).unsqueeze(-1).expand_as(
                    beam_seq_logprobs))

            beam_seq = torch.cat([beam_seq, selected_ix.unsqueeze(-1)], -1)  # beam_seq Nxbxl

            beam_logprobs_sum = beam_logprobs_sum.gather(1, beam_ix) + \
                                logprobs.reshape(batch_size, -1).gather(1, ix)
            assert (beam_logprobs_sum == ys).all()
            _tmp_beam_logprobs = unaug_logprobs[state_ix].reshape(batch_size, -1, vocab_size)
            beam_logprobs = unaug_logprobs.reshape(batch_size, -1, vocab_size).gather(1,
                                                                                      beam_ix.unsqueeze(-1).expand(-1,
                                                                                                                   -1,
                                                                                                                   vocab_size))  # NxbxV
            assert (_tmp_beam_logprobs == beam_logprobs).all()
            beam_seq_logprobs = torch.cat([
                beam_seq_logprobs,
                beam_logprobs.reshape(batch_size, -1, 1, vocab_size)], 2)

            new_state = [None for _ in state]
            for _ix in range(len(new_state)):
                #  copy over state in previous beam q to new beam at vix
                new_state[_ix] = state[_ix][:, state_ix]
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state

        # Start diverse_beam_search
        temperature = self.temperature # This should not affect beam search, but will affect dbs
        beam_size = self.beam_size
        group_size = self.group_size
        diversity_lambda = self.diversity_lambda
        decoding_constraint = self.decoding_constraint
        suppress_UNK = self.suppress_UNK
        length_penalty = penalty_builder(self.length_penalty)
        bdash = beam_size // group_size  # beam per group

        batch_size = init_logprobs.shape[0]
        device = init_logprobs.device
        # INITIALIZATIONS
        beam_seq_table = [torch.LongTensor(batch_size, bdash, 0).to(device) for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(batch_size, bdash, 0, self.vocab_size).to(device) for _ in
                                   range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(batch_size, bdash).to(device) for _ in range(group_size)]

        # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
        done_beams_table = [[[] for __ in range(group_size)] for _ in range(batch_size)]
        state_table = [[_.clone() for _ in init_state] for _ in range(group_size)]
        logprobs_table = [init_logprobs.clone() for _ in range(group_size)]
        # END INIT

        # Chunk elements in the args
        args = [patch_feats, mask]
        args = split_tensors(group_size, args)  #ßßßß For each arg, turn (Bbg)x... to (Bb)x(g)x...
        if self.__class__.__name__ == 'AttEnsemble':
            args = [[[args[j][i][k] for i in range(len(self.models))] for j in range(len(args))] for k in
                    range(group_size)]  # group_name, arg_name, model_name
        else:
            args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        for t in range(self.max_seq_length + group_size - 1):
            for divm in range(group_size):
                if t >= divm and t <= self.max_seq_length + divm - 1:
                    # add diversity
                    logprobs = logprobs_table[divm]
                    # suppress previous word
                    if decoding_constraint and t - divm > 0:
                        logprobs.scatter_(1, beam_seq_table[divm][:, :, t - divm - 1].reshape(-1, 1).to(device),
                                          float('-inf'))
                    # suppress UNK tokens in the decoding
                    if suppress_UNK and hasattr(self, 'vocab') and self.vocab[str(logprobs.size(1) - 1)] == 'UNK':
                        logprobs[:, logprobs.size(1) - 1] = logprobs[:, logprobs.size(1) - 1] - 1000
                        # diversity is added here
                    # the function directly modifies the logprobs values and hence, we need to return
                    # the unaugmented ones for sorting the candidates in the end. # for historical
                    # reasons :-)
                    logprobs, unaug_logprobs = add_diversity(beam_seq_table, logprobs, t, divm, diversity_lambda, bdash)

                    # infer new beams
                    beam_seq_table[divm], \
                    beam_seq_logprobs_table[divm], \
                    beam_logprobs_sum_table[divm], \
                    state_table[divm] = beam_step(logprobs,
                                                  unaug_logprobs,
                                                  bdash,
                                                  t - divm,
                                                  beam_seq_table[divm],
                                                  beam_seq_logprobs_table[divm],
                                                  beam_logprobs_sum_table[divm],
                                                  state_table[divm])
                    # if time's up... or if end token is reached then copy beams
                    for b in range(batch_size):
                        is_end = (beam_seq_table[divm][b, :, t - divm] == self.eos_idx)
                        assert beam_seq_table[divm].shape[-1] == t - divm + 1
                        if t == self.max_seq_length + divm - 1:
                            is_end.fill_(1)
                        for vix in range(bdash):
                            if is_end[vix]:
                                # final_beam = {
                                #     'seq': beam_seq_table[divm][b, vix].clone().cpu(),
                                #     'logps': beam_seq_logprobs_table[divm][b, vix].clone().cpu(),
                                #     'unaug_p': beam_seq_logprobs_table[divm][b, vix].cpu().sum().item(),
                                #     'p': beam_logprobs_sum_table[divm][b, vix].cpu().item(),
                                #     #'attn': attns[b][vix] if self.vis else None
                                # }
                                final_beam = {
                                    'seq': beam_seq_table[divm][b, vix].clone().cpu(),
                                    'logps': beam_seq_logprobs_table[divm][b, vix].clone().cpu(),
                                    'unaug_p': beam_seq_logprobs_table[divm][b, vix].sum().item(),
                                    'p': beam_logprobs_sum_table[divm][b, vix].item(),
                                    #'attn': attns[b][vix] if self.vis else None
                                }
                                final_beam['p'] = length_penalty(t - divm + 1, final_beam['p'])
                                done_beams_table[b][divm].append(final_beam)
                        beam_logprobs_sum_table[divm][b, is_end] -= 1000

                    # move the current group one step forward in time

                    it = beam_seq_table[divm][:, :, t - divm].reshape(-1)
                    logprobs_table[divm], state_table[divm] = model.core(it.cuda(),*(args[divm] + [state_table[divm]]))
                    # if self.vis:
                    #     attns = [attn.reshape(batch_size, -1, 1, *attn.shape[1:]) for attn in attns]
                    #     attns = torch.cat((attns[0], attns[1], attns[2]), dim=2).detach().cpu()

                    logprobs_table[divm] = F.log_softmax(logprobs_table[divm] / temperature, dim=-1)

        # all beams are sorted by their log-probabilities
        # if self.vis:
        #     attns = [attn.reshape(batch_size, -1, 1, *attn.shape[1:] ) for attn in attns]
        #     attns = torch.cat((attns[0], attns[1], attns[2]), dim=2)
        #     for b in range(batch_size):
        #         for i in range(group_size):
        #             for j in range(bdash):
        #                 print(len(done_beams_table[b][i]), attns[b].shape)
        #                 done_beams_table[b][i][j]['attn'] = attns[b][j]
        #             print(sorted(done_beams_table[b][i], key=lambda x: -x['p'])[:bdash][0].keys())
        #
        # print(done_beams_table[0][0][0].keys(), 111)
        done_beams_table = [[sorted(done_beams_table[b][i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
                            for b in range(batch_size)]
        # print(done_beams_table[0][0][0].keys(), 222)
        done_beams = [sum(_, []) for _ in done_beams_table]

        return done_beams