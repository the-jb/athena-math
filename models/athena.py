import torch
import torch.nn as nn
from models.attention import AttentionLayer, FeedForwardNorm
from thought_object import *
import itertools


class Athena(nn.Module):
    def __init__(
        self,
        max_depth,
        constants,
        lm_hidden_size,
        hidden_size,
        n_heads,
        ff_size,
        p_drop,
        ln,
        strength,
        chain: 0 | 1 | 2,
        goal: 0 | 1,
        ref: bool,
        reason: bool,
    ):
        super().__init__()
        self.max_depth = max_depth

        if hidden_size is None:
            self.scale = lambda x: x
            hidden_size = lm_hidden_size
        else:
            self.scale = nn.Sequential(
                nn.Linear(lm_hidden_size, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size, eps=1e-5)
            )
            n_heads = hidden_size * n_heads // lm_hidden_size
        if ff_size < hidden_size:
            ff_size = hidden_size

        self.constants = constants
        if constants:
            self.const_features = nn.Parameter(torch.randn(len(constants), hidden_size), requires_grad=True)
            self.act_const = nn.Sequential(
                nn.LayerNorm(hidden_size, eps=1e-5),
                nn.Linear(hidden_size, ff_size),
                nn.GELU(),
                nn.Linear(ff_size, hidden_size),
            )
        self.goal = goal
        self.strength_thd = strength
        self.embed_thought = EmbedThought(hidden_size, n_heads, ff_size, p_drop, ln)
        self.reason = ReasoningModule(hidden_size, n_heads, ff_size, p_drop, ln, reason and ref)
        self.answer = AnsweringModule(hidden_size, n_heads, ff_size, p_drop, ln, ref)
        self.chain = chain

    def forward(self, batch, features, threshold=0.5):
        batch_size = len(batch["input_ids"])

        batch_final_thought = []
        batch_final_loss, batch_reasoning_loss = 0.0, 0.0
        features = self.scale(features)
        batch_scores = []
        for batch_idx in range(batch_size):
            initial_thoughts = batch["initial_thoughts"][batch_idx]
            num_indices = batch["num_indices"][batch_idx]
            question_idx = batch["question_idx"][batch_idx]

            reasoning_feature = features[batch_idx, 0].unsqueeze(0)
            if self.goal == 0:
                goal_feature = features[batch_idx, question_idx].unsqueeze(0)
            else:
                question_indices = batch["question_indices"][batch_idx]
                goal_feature = features[batch_idx, question_indices]

            initial_indices = [num_indices[e.idx] for e in initial_thoughts if isinstance(e, NumberObject)]
            initial_embedding = features[batch_idx, initial_indices]
            if self.constants:
                const_indices = [self.constants.index(e) for e in initial_thoughts if isinstance(e, ConstObject)]
                const_features = self.act_const(self.const_features)
                initial_embedding = torch.cat((initial_embedding, const_features[const_indices]))

            if self.training:
                embeddings, thoughts, reasoning_outputs, reasoning_loss = self.supervise_thoughts(
                    initial_embedding,
                    reasoning_feature,
                    batch["label_thoughts"][batch_idx],
                    batch["label_thought_indices"][batch_idx],
                    batch["label_dd"][batch_idx],
                    batch["label_dd_indices"][batch_idx],
                )
                batch_reasoning_loss += reasoning_loss

                final_output, final_loss = self.answer(embeddings, goal_feature, label=batch["label_final"][batch_idx])
                batch_final_loss += final_loss
            else:
                embeddings, thoughts, reasoning_outputs = self.predict_thoughts(
                    initial_embedding, initial_thoughts, reasoning_feature, goal_feature, threshold
                )
                final_output, strength, _ = self.answer(embeddings, goal_feature)
                _, _, score = self.answer(embeddings, features[batch_idx])
                score = score.cpu().numpy()[0]
                score = {t.expr: s for t, s in zip(thoughts, score)}
                batch_scores.append(score)

            final_idx = final_output.argmax(-1)
            batch_final_thought.append(thoughts[final_idx])

        if self.training:
            n_dds = sum(batch["n_dds"])
            n_thoughts = sum(batch["n_thoughts"])
            loss = (batch_final_loss + batch_reasoning_loss) / (n_dds + n_thoughts)
            return batch_final_thought, loss
        return batch_final_thought, batch_scores

    def supervise_thoughts(
        self, initial_embeddings, reasoning_feature, label_thoughts, label_thought_indices, label_dds, label_dd_indices
    ):
        total_loss = 0
        collected_thoughts = []
        outputs = []

        embeddings = None

        for depth, (thoughts, thought_indices, label_dd, dd_indices) in enumerate(
            zip(label_thoughts, label_thought_indices, label_dds, label_dd_indices)
        ):
            expanded_embeddings = initial_embeddings if depth == 0 else self.embed_thought(embeddings, thought_indices)

            thought_context, output, loss = self.reason(expanded_embeddings, reasoning_feature, label=label_dd)

            if len(dd_indices) > 0:
                outputs.append(output[dd_indices])
                expanded_embeddings = expanded_embeddings[dd_indices]
                thought_context = thought_context[dd_indices]
                embeddings = (
                    expanded_embeddings if embeddings is None else torch.cat((embeddings, expanded_embeddings), dim=0)
                )
                collected_thoughts += [thoughts[i] for i in dd_indices]
                if self.chain == 1:
                    reasoning_feature = torch.cat((reasoning_feature, thought_context))
                elif self.chain == 2:
                    reasoning_feature = thought_context

            total_loss += loss
        outputs = torch.cat(outputs)
        return embeddings, collected_thoughts, outputs, total_loss

    def predict_thoughts(self, initial_embeddings, initial_thoughts, reasoning_feature, goal_feature, threshold):
        outputs = []
        embeddings = None
        expander = ThoughtExpander(initial_thoughts, self.max_depth)

        for expanded_thoughts, expanded_indices in expander:
            expanded_embeddings = (
                initial_embeddings if expanded_indices is None else self.embed_thought(embeddings, expanded_indices)
            )

            thought_context, output = self.reason(expanded_embeddings, reasoning_feature)
            accepted_thoughts = output.ge(threshold)

            if accepted_thoughts.count_nonzero() == 0:
                if expanded_indices is None:
                    outputs.append(output)
                    embeddings = expanded_embeddings
                    expander.collect(expanded_thoughts)
                    if self.chain == 1:
                        reasoning_feature = torch.cat((reasoning_feature, thought_context))
                    elif self.chain == 2:
                        reasoning_feature = thought_context
            else:
                outputs.append(output[accepted_thoughts])
                expanded_embeddings = expanded_embeddings[accepted_thoughts]
                thought_context = thought_context[accepted_thoughts]
                embeddings = (
                    expanded_embeddings if embeddings is None else torch.cat((embeddings, expanded_embeddings), dim=0)
                )
                expander.collect([thought for thought, s in zip(expanded_thoughts, accepted_thoughts) if s])
                if self.chain == 1:
                    reasoning_feature = torch.cat((reasoning_feature, thought_context))
                elif self.chain == 2:
                    reasoning_feature = thought_context

            final_output, strength, _ = self.answer(embeddings, goal_feature)
            if strength >= self.strength_thd:
                break

        outputs = torch.cat(outputs)
        return embeddings, expander.thoughts, outputs


class EmbedThought(nn.Module):
    def __init__(self, hidden_size, n_heads, ff_size, p_drop, ln):
        super().__init__()
        self.embed = nn.ModuleDict(
            {op: Transform(hidden_size, ff_size, p_drop, ln) for op in TransformObject.OPS}
            | {op: Aggregate(hidden_size, n_heads, ff_size, p_drop, ln) for op in AggregateObject.OPS}
        )

    def forward(self, embeddings, indices):
        x = []
        for op, group in itertools.groupby(indices, key=lambda item: item[0]):
            op_indices = [i for _, i in group]
            x.append(self.embed[op](embeddings, op_indices))

        x = torch.cat(x)
        return x


class Correlation(nn.Module):
    def __init__(self, hidden_size, n_heads, ff_size, p_drop, ln, ref):
        super().__init__()
        self.ffn = FeedForwardNorm(hidden_size, ff_size, p_drop, ln)
        self.attention = AttentionLayer(hidden_size, n_heads, ff_size, p_drop, ln)
        self.proj = nn.Linear(hidden_size, 1)
        self.loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.ref = ref

    def forward(self, embeddings, features):
        x = self.ffn(embeddings.unsqueeze(0))
        if self.ref:
            x, score = self.attention(x, features.unsqueeze(0), mask=None)
        else:
            score = None
        x = x.squeeze(0)
        p = self.proj(x).squeeze(-1)
        return x, p, score


class ReasoningModule(Correlation):
    def forward(self, embeddings, features, label=None):
        x, p, _ = super().forward(embeddings, features)
        if label is not None:
            loss = self.loss(p, label)
            return x, p.sigmoid(), loss
        return x, p.sigmoid()


class AnsweringModule(Correlation):
    def forward(self, embeddings, features, label=None):
        _, p, score = super().forward(embeddings, features)

        if label is not None:
            target = torch.zeros_like(p)
            target[label] = 1

            loss = self.loss(p, target)
            return p, loss
        strength = p.sigmoid().max()
        return p, strength, score


class Transform(nn.Module):
    def __init__(self, hidden_size, ff_size, p_drop, ln):
        super().__init__()
        self.ffn = FeedForwardNorm(hidden_size, ff_size, p_drop, ln)

    def forward(self, embedding, indices):
        x = self.ffn(embedding[indices])
        return x


class Aggregate(nn.Module):
    def __init__(self, hidden_size, n_heads, ff_size, p_drop, ln):
        super().__init__()
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_size = hidden_size // n_heads
        self.scale = self.head_size**-0.5

        self.dense = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(p_drop))
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNorm(hidden_size, ff_size, p_drop, ln)
        self.ln = ln

    def apply_qkv(self, embeddings, indices):
        q = self.q(embeddings).view(-1, self.n_heads, self.head_size).permute(1, 0, 2)
        k = self.k(embeddings).view(-1, self.n_heads, self.head_size).permute(1, 2, 0)
        v = self.v(embeddings).view(-1, self.n_heads, self.head_size).permute(1, 0, 2)[:, indices]
        return q, k, v

    def attention(self, embeddings, indices):
        rows = [((i, i), (j, j)) for i, j in indices]
        cols = [((i, j), (i, j)) for i, j in indices]
        q, k, v = self.apply_qkv(embeddings, indices)

        x = q @ k
        x = (self.scale * x)[:, rows, cols]
        x = torch.softmax(x, dim=-1)
        x = x @ v
        x = x.sum(dim=-2)
        x = x.permute(1, 0, 2).reshape(-1, self.hidden_size)
        return x

    def forward(self, embeddings, indices):
        if self.ln == "pre":
            x = self.norm(embeddings)
            x = self.attention(x, indices)
            x = self.dense(x)
            x += embeddings[indices, :].sum(dim=1)
        else:  # self.ln == "post":
            x = self.attention(embeddings, indices)
            x = self.dense(x)
            x += embeddings[indices, :].sum(dim=1)
            x = self.norm(x)

        x = self.ffn(x)
        return x


class OrderedAggr(Aggregate):
    def __init__(self, hidden_size, n_heads, ff_size, p_drop, ln):
        super().__init__(hidden_size, n_heads, ff_size, p_drop, ln)
        self.transform = FeedForwardNorm(hidden_size, ff_size, p_drop, ln)

    def apply_qkv(self, embeddings, indices):
        t = self.transform(embeddings)
        e = torch.stack((embeddings, t), dim=1)

        q = self.q(embeddings).view(-1, self.n_heads, self.head_size).permute(1, 0, 2)
        k = self.k(t).view(-1, self.n_heads, self.head_size).permute(1, 2, 0)
        v = self.v(e).view(-1, 2, self.n_heads, self.head_size).permute(2, 0, 1, 3)[:, indices, (0, 1)]
        return q, k, v
