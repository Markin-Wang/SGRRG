import torch
from .utils import split_tensors, repeat_tensors, penalty_builder
import torch.nn.functional as F


class BeamSearch:
    def __init__(self, config):
        super().__init__()
        # self.model = pl_module
        self.beam_size = config['beam_size']
        self.sample_method = config['sample_method']

        self.max_seq_length = config['max_seq_length']

        self.vocab_size = config['vocab_size']
        self.eos_idx = config['eos_idx']
        self.pad_idx = config['pad_idx']
        self.bos_idx = config['bos_idx']
        self.sgade = config['sgade']
        self.use_sg = config['use_sg']

    def set_tokenidx(self, tokenizer):
        self.pad_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.bos_idx = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.eos_idx = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        self.sep_idx = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        self.vocab_size = tokenizer.vocab_size

    def caption_test_step(self, model, batch_dict):
        captions = []
        max_len = self.max_seq_length
        if not model.training:
            # from fairseq
            image_embeds = batch_dict['encoded_img_feats']
            sg_embeds, sg_masks, bs_ids = batch_dict['sg_embeds'], batch_dict['sg_masks'], batch_dict['bs_ids']
            bs = image_embeds.size(0)
            text_ids = torch.ones((bs, max_len), device=image_embeds.device, dtype=torch.long)
            text_ids[:, 0] = self.bos_idx
            text_ids[:, 1:] = self.pad_idx
            search_size = bs * self.beam_size
            end_seq = torch.zeros_like(text_ids[:, 0])
            end_seq = end_seq > 0
            self_masks = torch.zeros((bs, max_len), device=image_embeds.device)
            self_masks[:, 0] = 1

            text_feats = model.infer(text_ids, image_embeds, self_masks, None, sg_embeds, sg_masks, bs_ids)

            self_masks = self_masks.view(bs, 1, -1).repeat(1, self.beam_size, 1).view(search_size, -1)

            for i in range(max_len - 1):
                self_masks[:, i] = 1
                if i != 0:
                    text_feats = model.infer(text_ids, image_embeds, self_masks, None, sg_embeds, sg_masks, bs_ids)

                mlm_logits = model.rrg_head(text_feats[:, i: i + 1])

                mlm_logits = torch.log_softmax(mlm_logits, dim=-1)
                if i == 0:
                    tgt_prev_tokens = mlm_logits.argsort(descending=True, dim=-1)[:, :, :self.beam_size]
                    head_logp = mlm_logits.gather(dim=-1, index=tgt_prev_tokens)
                    tgt_prev_tokens = tgt_prev_tokens.permute(0, 2, 1).reshape(search_size, 1).contiguous()
                    head_logp = head_logp.view(search_size, 1)
                    head_lengths = torch.ones_like(head_logp)
                    text_ids = text_ids.view(bs, 1, -1).repeat(1, self.beam_size, 1).view(search_size, -1)
                    end_seq = end_seq.view(bs, 1, -1).repeat(1, self.beam_size, 1).view(search_size, 1)

                    scores = torch.zeros_like(end_seq)
                    padded = torch.full((search_size, 1), 1, dtype=torch.long, device=text_ids.device)

                    hs = image_embeds.size(-1)
                    image_embeds = image_embeds.view(bs, 1, -1, hs).repeat(1, self.beam_size, 1, 1).view(search_size,
                                                                                                         -1, hs)
                    if sg_embeds is not None and self.sgade:
                        sg_embeds = sg_embeds.view(bs, 1, -1, hs).repeat(1, self.beam_size, 1, 1).view(search_size, -1,
                                                                                                       hs)
                        sg_masks = sg_masks.view(bs, 1, -1, sg_masks.shape[-1]).repeat(1, self.beam_size, 1, 1).view(
                            search_size, -1, sg_masks.shape[-1])
                        bs_ids = bs_ids.view(bs_ids.size(0), 1).repeat(1, self.beam_size) * self.beam_size
                        shift = torch.arange(0, self.beam_size).view(-1, self.beam_size).repeat(bs_ids.size(0), 1).to(
                            bs_ids.device)
                        bs_ids = (bs_ids + shift).view(-1)

                    text_ids[:, i + 1] = tgt_prev_tokens.view(-1)
                    # end_seq = (tgt_prev_tokens == tokenizer.sep_token_id) | (tgt_prev_tokens == tokenizer.pad_token_id)
                    end_seq = tgt_prev_tokens == self.pad_idx

                else:
                    decoder_lengths = 1.0 - end_seq.to(mlm_logits.dtype)
                    mlm_logits = mlm_logits * decoder_lengths[:, :, None]
                    mlm_logits = mlm_logits + head_logp[:, :, None]
                    mlm_logits = mlm_logits.view(bs, self.beam_size, 1, -1).permute(0, 2, 1, 3)
                    vocab_size = mlm_logits.size(3)
                    decoder_lengths = decoder_lengths + head_lengths
                    decoder_lengths = decoder_lengths.view(bs, self.beam_size, 1).permute(0, 2, 1)
                    decoder_normed_logp = (
                        (mlm_logits / (decoder_lengths[:, :, :, None] + 1e-9)).contiguous().view(bs, 1, -1)
                    )
                    decoder_logp = mlm_logits.contiguous().view(bs, 1, -1)
                    top_idx = decoder_normed_logp.argsort(descending=True, dim=-1)[:, :, :self.beam_size]
                    top_logp = decoder_logp.gather(dim=-1, index=top_idx)
                    top_tokens = top_idx % vocab_size
                    top_prev_idx = torch.div(top_idx, vocab_size, rounding_mode='floor')
                    # top_prev_idx = top_idx // vocab_size
                    top_prev_idx += torch.arange(bs, dtype=torch.long, device=mlm_logits.device)[:, None,
                                    None] * self.beam_size

                    top_prev_idx = top_prev_idx.permute(0, 2, 1)
                    top_prev_idx = top_prev_idx.contiguous().view(search_size, 1)
                    top_logp = top_logp.permute(0, 2, 1)
                    head_logp = top_logp.contiguous().view(search_size, 1)
                    top_lengths = decoder_lengths.permute(0, 2, 1)
                    head_lengths = top_lengths.contiguous().view(search_size, 1)
                    top_tokens = top_tokens.permute(0, 2, 1)
                    top_tokens = top_tokens.contiguous().view(search_size, 1)
                    prev_ended = end_seq.gather(dim=0, index=top_prev_idx)
                    tgt_prev_tokens = (1 - prev_ended.to(torch.long)) * top_tokens + prev_ended.to(torch.long) * padded
                    t_size = text_ids.size(1)
                    top_decoded_idx = top_prev_idx.repeat(1, t_size)
                    text_ids = text_ids.gather(dim=0, index=top_decoded_idx)
                    text_ids[:, i + 1] = tgt_prev_tokens.view(-1)
                    # end_seq = (tgt_prev_tokens == tokenizer.sep_token_id) | (tgt_prev_tokens == tokenizer.pad_token_id)
                    end_seq = tgt_prev_tokens == self.pad_idx

                    if torch.sum(end_seq) == len(end_seq):
                        break

            text_ids = text_ids.view(bs, self.beam_size, -1)[:, 0, 1:]
            text_ids = text_ids.contiguous()
            # for text_id in text_ids:
            #     captions.append(tokenizer.decode(text_id).replace(tokenizer.pad_token, ""))
            # captions = tokenizer.batch_decode(text_ids, skip_special_tokens=True)
        # return {"image_ids": batch["iid"], "captions": captions, "gts": gts}
        return {'preds': text_ids}

    # def beam_search_simplified(self, model, init_state, init_logprobs,patch_feats=None,mask=None):
    #
    #     # function computes the similarity score to be augmented
    #     # does one step of classical beam search
    #
    #     def beam_step(logprobs, beam_size, t, beam_seq, beam_seq_logprobs,
    #                   beam_logprobs_sum, state):
    #         # INPUTS:
    #         # logprobs: probabilities augmented after diversity N*bxV
    #         # beam_size: obvious
    #         # t        : time instant
    #         # beam_seq : tensor contanining the beams
    #         # beam_seq_logprobs: tensor contanining the beam logprobs
    #         # beam_logprobs_sum: tensor contanining joint logprobs
    #         # OUPUTS:
    #         # beam_seq : tensor containing the word indices of the decoded captions Nxbxl
    #         # beam_seq_logprobs : log-probability of each decision made, NxbxlxV
    #         # beam_logprobs_sum : joint log-probability of each beam Nxb
    #
    #         batch_size = beam_logprobs_sum.shape[0]
    #         vocab_size = logprobs.shape[-1]
    #         logprobs = logprobs.reshape(batch_size, -1, vocab_size)  # NxbxV
    #         if t == 0:
    #             assert logprobs.shape[1] == 1
    #             beam_logprobs_sum = beam_logprobs_sum[:, :1]
    #         candidate_logprobs = beam_logprobs_sum.unsqueeze(-1) + logprobs  # beam_logprobs_sum Nxb logprobs is NxbxV
    #         # ys, ix = torch.sort(candidate_logprobs.reshape(candidate_logprobs.shape[0], -1), -1, True)
    #         # ys, ix = ys[:, :beam_size], ix[:, :beam_size]
    #         ys, ix = torch.topk(candidate_logprobs.reshape(candidate_logprobs.shape[0], -1), beam_size, -1, True)
    #         # print(ys, ix)
    #         # exit()
    #         # beam_ix = ix // vocab_size  # Nxb which beam
    #         # pytorch version difference
    #         beam_ix = torch.div(ix, vocab_size, rounding_mode='floor')
    #         selected_ix = ix % vocab_size  # Nxb # which world
    #         state_ix = (beam_ix + torch.arange(batch_size).type_as(beam_ix).unsqueeze(-1) * logprobs.shape[1]).reshape(
    #             -1)  # N*b which in Nxb beams
    #
    #         if t > 0:
    #             # gather according to beam_ix
    #
    #             beam_seq = beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq))
    #
    #             beam_seq_logprobs = beam_seq_logprobs.gather(1, beam_ix.unsqueeze(-1).unsqueeze(-1).expand_as(
    #                 beam_seq_logprobs))
    #
    #         beam_seq = torch.cat([beam_seq, selected_ix.unsqueeze(-1)], -1)  # beam_seq Nxbxl
    #
    #         beam_logprobs_sum = beam_logprobs_sum.gather(1, beam_ix) + \
    #                             logprobs.reshape(batch_size, -1).gather(1, ix)
    #         assert (beam_logprobs_sum == ys).all()
    #
    #
    #
    #         new_state = [None for _ in state]
    #         for _ix in range(len(new_state)):
    #             #  copy over state in previous beam q to new beam at vix
    #             new_state[_ix] = state[_ix][:, state_ix]
    #         state = new_state
    #         return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state
    #
    #     # Start diverse_beam_search
    #     temperature = self.temperature # This should not affect beam search, but will affect dbs
    #     beam_size = self.beam_size
    #     group_size = self.group_size
    #     diversity_lambda = self.diversity_lambda
    #     decoding_constraint = self.decoding_constraint
    #     suppress_UNK = self.suppress_UNK
    #     length_penalty = penalty_builder(self.length_penalty)
    #     bdash = beam_size // group_size  # beam per group
    #
    #     batch_size = init_logprobs.shape[0]
    #     device = init_logprobs.device
    #     # INITIALIZATIONS
    #     beam_seq_table = torch.LongTensor(batch_size, bdash, 0).to(device)
    #     beam_seq_logprobs = torch.FloatTensor(batch_size, bdash, 0, self.vocab_size).to(device)
    #     beam_logprobs_sum = torch.zeros(batch_size, bdash).to(device)
    #
    #     # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
    #     done_beams = [[]  for _ in range(batch_size)]
    #     state_table = [_.clone() for _ in init_state]
    #     logprobs_table = init_logprobs.clone()
    #     # END INIT
    #
    #     # Chunk elements in the args
    #     args = [patch_feats, mask]
    #     #args = split_tensors(group_size, args)  #ßßßß For each arg, turn (Bbg)x... to (Bb)x(g)x...
    #
    #
    #     for t in range(self.max_seq_length - 1):
    #                 # add diversity
    #         # the function directly modifies the logprobs values and hence, we need to return
    #         # the unaugmented ones for sorting the candidates in the end. # for historical
    #         # reasons :-)
    #
    #         # infer new beams
    #         beam_seq_table[divm], \
    #         beam_seq_logprobs_table[divm], \
    #         beam_logprobs_sum_table[divm], \
    #         state_table[divm] = beam_step(logprobs,
    #                                       bdash,
    #                                       t - divm,
    #                                       beam_seq_table[divm],
    #                                       beam_seq_logprobs_table[divm],
    #                                       beam_logprobs_sum_table[divm],
    #                                       state_table[divm])
    #
    #         logprobs = logprobs.reshape(batch_size, -1, vocab_size)  # NxbxV
    #         if t == 0:
    #             assert logprobs.shape[1] == 1
    #             beam_logprobs_sum = beam_logprobs_sum[:, :1]
    #             # beam_logprobs_sum Nxb logprobs is NxbxV
    #         candidate_logprobs = beam_logprobs_sum.unsqueeze(-1) + logprobs
    #
    #
    #         ys, ix = torch.topk(candidate_logprobs.reshape(candidate_logprobs.shape[0], -1), beam_size, -1,
    #                             True)
    #
    #         beam_ix = torch.div(ix, vocab_size, rounding_mode='floor')
    #         selected_ix = ix % vocab_size  # Nxb # which world
    #         state_ix = (beam_ix + torch.arange(batch_size).type_as(beam_ix).unsqueeze(-1) * logprobs.shape[
    #             1]).reshape(
    #             -1)  # N*b which in Nxb beams
    #
    #         if t > 0:
    #             # gather according to beam_ix
    #
    #             beam_seq = beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq))
    #
    #             beam_seq_logprobs = beam_seq_logprobs.gather(1, beam_ix.unsqueeze(-1).unsqueeze(-1).expand_as(
    #                 beam_seq_logprobs))
    #
    #         beam_seq = torch.cat([beam_seq, selected_ix.unsqueeze(-1)], -1)  # beam_seq Nxbxl
    #
    #         beam_logprobs_sum = beam_logprobs_sum.gather(1, beam_ix) + \
    #                             logprobs.reshape(batch_size, -1).gather(1, ix)
    #         assert (beam_logprobs_sum == ys).all()
    #
    #         new_state = [None for _ in state]
    #         for _ix in range(len(new_state)):
    #             #  copy over state in previous beam q to new beam at vix
    #             new_state[_ix] = state[_ix][:, state_ix]
    #         state = new_state
    #
    #         # if time's up... or if end token is reached then copy beams
    #         beam_seq = beam_seq.view(batch_size*beam_size,-1)
    #         is_end = beam_seq[:,t] == self.eos_idx
    #         if torch.sum(is_end) == len(is_end):
    #             break
    #         for b in range(batch_size):
    #             is_end = beam_seq[b, :, t] == self.eos_idx
    #
    #             if t == self.max_seq_length - 1:
    #                 is_end.fill_(1)
    #             for vix in range(bdash):
    #                 if is_end[vix]:
    #                     # final_beam = {
    #                     #     'seq': beam_seq_table[divm][b, vix].clone().cpu(),
    #                     #     'logps': beam_seq_logprobs_table[divm][b, vix].clone().cpu(),
    #                     #     'unaug_p': beam_seq_logprobs_table[divm][b, vix].cpu().sum().item(),
    #                     #     'p': beam_logprobs_sum_table[divm][b, vix].cpu().item(),
    #                     #     #'attn': attns[b][vix] if self.vis else None
    #                     # }
    #                     final_beam = {
    #                         'seq': beam_seq_table[divm][b, vix].clone().cpu(),
    #                         'logps': beam_seq_logprobs_table[divm][b, vix].clone().cpu(),
    #                         'unaug_p': beam_seq_logprobs_table[divm][b, vix].sum().item(),
    #                         'p': beam_logprobs_sum_table[divm][b, vix].item(),
    #                         #'attn': attns[b][vix] if self.vis else None
    #                     }
    #                     final_beam['p'] = length_penalty(t - divm + 1, final_beam['p'])
    #                     done_beams_table[b][divm].append(final_beam)
    #             beam_logprobs_sum_table[divm][b, is_end] -= 1000
    #
    #                 # move the current group one step forward in time
    #
    #                 it = beam_seq_table[divm][:, :, t - divm].reshape(-1)
    #                 logprobs_table[divm], state_table[divm] = model.core(it.to(device),*(args[divm] + [state_table[divm]]))
    #                 # if self.vis:
    #                 #     attns = [attn.reshape(batch_size, -1, 1, *attn.shape[1:]) for attn in attns]
    #                 #     attns = torch.cat((attns[0], attns[1], attns[2]), dim=2).detach().cpu()
    #
    #                 logprobs_table[divm] = F.log_softmax(logprobs_table[divm] / temperature, dim=-1)
    #
    #     # all beams are sorted by their log-probabilities
    #     # if self.vis:
    #     #     attns = [attn.reshape(batch_size, -1, 1, *attn.shape[1:] ) for attn in attns]
    #     #     attns = torch.cat((attns[0], attns[1], attns[2]), dim=2)
    #     #     for b in range(batch_size):
    #     #         for i in range(group_size):
    #     #             for j in range(bdash):
    #     #                 print(len(done_beams_table[b][i]), attns[b].shape)
    #     #                 done_beams_table[b][i][j]['attn'] = attns[b][j]
    #     #             print(sorted(done_beams_table[b][i], key=lambda x: -x['p'])[:bdash][0].keys())
    #     #
    #     # print(done_beams_table[0][0][0].keys(), 111)
    #     done_beams_table = [[sorted(done_beams_table[b][i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
    #                         for b in range(batch_size)]
    #     # print(done_beams_table[0][0][0].keys(), 222)
    #     done_beams = [sum(_, []) for _ in done_beams_table]
    #
    #     return done_beams
