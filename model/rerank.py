import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import dgl
from dgl import function as fn

__all__ = ["KBGATLayer", "GATEncoder", "InteractEAdapter", "GraphEnhancer"]

class KBGATLayer(nn.Module):
    """
    Knowledge Graph Attention Layer based on Nathani et al. (ACL 2019).
    Implements asymmetric attention: Score = LeakyReLU(a^T [h_i || h_j || r_k])
    """
    def __init__(self, in_dim, out_dim, rel_dim, num_heads=4, alpha=0.2, dropout=0.3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rel_dim = rel_dim
        self.num_heads = num_heads
        
        self.attn_vec = nn.Parameter(torch.empty(num_heads, 2 * in_dim + rel_dim))
        nn.init.xavier_uniform_(self.attn_vec)
        
        self.W_msg = nn.Linear(in_dim + rel_dim, out_dim * num_heads, bias=False)
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, n_feats, r_feats, etypes):
        with g.local_scope():
            g.ndata['h'] = n_feats
            g.edata['r'] = r_feats[etypes]
            
            def edge_attention(edges):
                z = torch.cat([edges.src['h'], edges.dst['h'], edges.data['r']], dim=-1)
                z_mh = z.unsqueeze(1).expand(-1, self.num_heads, -1)
                score = (z_mh * self.attn_vec).sum(dim=-1)
                return {'e': self.leaky_relu(score)}

            g.apply_edges(edge_attention)
            g.edata['alpha'] = dgl.ops.edge_softmax(g, g.edata['e'])
            g.edata['alpha'] = self.dropout(g.edata['alpha'])
            
            msg_input = torch.cat([g.ndata['h'][g.edges()[0]], g.edata['r']], dim=-1)
            messages = self.W_msg(msg_input).view(-1, self.num_heads, self.out_dim)
            g.edata['m'] = messages * g.edata['alpha'].unsqueeze(-1)
            
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h_new'))
            
            h_new = g.ndata['h_new'].view(-1, self.num_heads * self.out_dim)
            return h_new

class GATEncoder(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, rel_dim, num_layers=1, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        last_h_dim = out_dim // num_heads
        
        if num_layers == 1:
            self.layers.append(KBGATLayer(in_dim, last_h_dim, rel_dim, num_heads))
        else:
            self.layers.append(KBGATLayer(in_dim, h_dim, rel_dim, num_heads))
            for _ in range(num_layers - 2):
                self.layers.append(KBGATLayer(h_dim * num_heads, h_dim, rel_dim, num_heads))
            self.layers.append(KBGATLayer(h_dim * num_heads, last_h_dim, rel_dim, num_heads))

    def forward(self, g, n_feats, r_feats, etypes):
        h = n_feats
        for layer in self.layers:
            h = layer(g, h, r_feats, etypes)
            h = F.elu(h)
        return h

class InteractEAdapter(nn.Module):
    """
    Paper-Aligned InteractE Adapter (AAAI 2020).
    Uses Interleaving/Chequerboard reshaping and Circular Convolution.
    """
    def __init__(self, input_size, output_size, adapter_size=1024, num_filters=96, kernel_size=9, dropout=0.2):
        super().__init__()
        import math
        self.k_h = int(math.sqrt(input_size))
        while input_size % self.k_h != 0:
            self.k_h -= 1
        self.k_w = input_size // self.k_h
        
        
        self.num_filt = num_filters
        self.ker_sz = kernel_size
        
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.bn2 = nn.BatchNorm1d(output_size)
        
        self.inp_drop = nn.Dropout(dropout)
        self.feat_drop = nn.Dropout2d(dropout)
        self.hid_drop = nn.Dropout(dropout)
        
        self.conv_filt = nn.Parameter(torch.zeros(num_filters, 1, kernel_size, kernel_size))
        nn.init.xavier_normal_(self.conv_filt)
        
        # Multi-layer Projection
        flat_sz = num_filters * self.k_h * (2 * self.k_w)
        self.fc1 = nn.Linear(flat_sz, adapter_size)
        self.fc2 = nn.Linear(adapter_size, output_size)

    def circular_padding(self, x, pad):
        upper_pad = x[..., -pad:, :]
        lower_pad = x[..., :pad, :]
        return torch.cat([upper_pad, x, lower_pad], dim=2)

    def forward(self, h_emb, c_emb):
        # h_emb: Identity (B, 256), c_emb: Context (B, 256)
        batch_size = h_emb.size(0)
        
        # --- Paper Logic: Interleaving ---
        # Instead of [h || c], we interleave: [h1, c1, h2, c2, ...]
        # This is equivalent to a chequerboard pattern when reshaped
        stacked = torch.stack([h_emb, c_emb], dim=2) # (B, 256, 2)
        interleaved = stacked.view(batch_size, -1) # (B, 512)
        
        # Reshape to 2D: (B, 1, 16, 32)
        x = interleaved.view(batch_size, 1, self.k_h, 2 * self.k_w)
        x = self.bn0(x)
        x = self.inp_drop(x)
        
        # Circular Convolution
        pad = self.ker_sz//2
        x = self.circular_padding(x, pad)
        x = F.pad(x, [pad, pad, 0, 0])
        x = F.conv2d(x, self.conv_filt, padding=0)
        x = self.bn1(x)
        x = F.silu(x)
        x = self.feat_drop(x)
        
        # Flatten and Project (2-Layer MLP)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = F.silu(x)
        x = self.hid_drop(x)
        x = self.fc2(x)
        x = self.hid_drop(x)
        return x

class GraphEnhancer(nn.Module):
    def __init__(self, kge_embedding, input_size, rel_input, gnn_hidden_dim, gnn_num_hidden_layers, adapter_size, output_size, hidden_act='silu', freeze_ent=False):
        super().__init__()
        self.ent_embeddings = nn.Embedding.from_pretrained(kge_embedding, freeze=freeze_ent)
        self.input_size = input_size
        
        if isinstance(rel_input, torch.Tensor):
            self.rel_embeddings = nn.Embedding.from_pretrained(rel_input, freeze=freeze_ent)
            self.rel_dim = rel_input.shape[1]
        else:
            self.rel_dim = 128
            self.rel_embeddings = nn.Embedding(rel_input, self.rel_dim)
            nn.init.xavier_uniform_(self.rel_embeddings.weight)

        self.gat = GATEncoder(input_size, gnn_hidden_dim, input_size, self.rel_dim, gnn_num_hidden_layers)
        self.adapter = InteractEAdapter(input_size, output_size, adapter_size=adapter_size)

    def forward(self, query_ids, entity_ids, subgraph):
        device = query_ids.device
        batch_size = query_ids.size(0)
        num_candidates = entity_ids.size(1)

        # 1. Handle No Subgraph Case (Fully Batched)
        if (subgraph is None):
            flat_qe = torch.cat([query_ids.view(-1, 1), entity_ids], dim=1) # (B, K+1)
            flat_ids = flat_qe.view(-1) #(B*(K+1), )
            base_vecs = self.ent_embeddings(flat_ids) # (B*(K+1), 256)
            # Self-interleave for no-subgraph
            out_vecs = self.adapter(base_vecs, base_vecs) # (B*(K+1), output_size)
            out_vecs = out_vecs.view(batch_size, (num_candidates + 1), -1)
            return out_vecs[:, 0, :], out_vecs[:, 1:, :].reshape(batch_size * num_candidates, -1)

        # 2. Handle Subgraph Case
        # all_q_id_embs = []
        # all_q_ctx_embs = []
        # all_e_id_embs = []
        # all_e_ctx_embs = []

        # for i in range(batch_size):
        #     edges_list = subgraph[i]
            
        #     # Case: Subgraph too small
        #     if len(edges_list) <= 5:
        #         q_id_emb = self.ent_embeddings(query_ids[i].unsqueeze(0))
        #         all_q_id_embs.append(q_id_emb)
        #         all_q_ctx_embs.append(q_id_emb)
                
        #         e_id_embs = self.ent_embeddings(entity_ids[i])
        #         all_e_id_embs.append(e_id_embs)
        #         all_e_ctx_embs.append(e_id_embs)
        #         continue

        #     # Build Graph
        #     edges_arr = np.array(edges_list)
        #     src, r, dst = edges_arr[:, 0], edges_arr[:, 1], edges_arr[:, 2]
        #     node_ids_sub = np.unique(np.concatenate([src, dst]))
        #     node_to_idx = {old: idx for idx, old in enumerate(node_ids_sub)}
        #     g = dgl.graph(([node_to_idx[s] for s in src], [node_to_idx[d] for d in dst]), num_nodes=len(node_ids_sub)).to(device)
        #     etypes = torch.LongTensor(r).to(device)
            
        #     # GAT Refinement
        #     base_emb_sub = self.ent_embeddings(torch.LongTensor(node_ids_sub).to(device))
        #     gat_emb_sub = self.gat(g, base_emb_sub, self.rel_embeddings.weight, etypes)

        #     # Collect Query
        #     qid = query_ids[i].item()
        #     q_id_emb = self.ent_embeddings(query_ids[i].unsqueeze(0))
        #     q_ctx_emb = gat_emb_sub[node_to_idx[qid]].unsqueeze(0) if qid in node_to_idx else q_id_emb
        #     all_q_id_embs.append(q_id_emb)
        #     all_q_ctx_embs.append(q_ctx_emb)

        #     # Collect Candidates
        #     e_id_embs_batch = self.ent_embeddings(entity_ids[i])
        #     all_e_id_embs.append(e_id_embs_batch)
            
        #     e_ctx_list = []
        #     for eid in entity_ids[i].tolist():
        #         ctx = gat_emb_sub[node_to_idx[eid]].unsqueeze(0) if eid in node_to_idx else self.ent_embeddings(torch.LongTensor([eid]).to(device))
        #         e_ctx_list.append(ctx)
        #     all_e_ctx_embs.append(torch.cat(e_ctx_list, dim=0))


        # --- STEP 1: PREPARE AND BATCH GRAPHS ---
        dgl_graphs = []
        all_node_ids = []
        all_etypes = []
        node_mappings = {} # Map batch index 'i' to its node_to_idx dictionary

        for i in range(batch_size):
            edges_list = subgraph[i]
            
            # Case: Subgraph too small
            if len(edges_list) <= 5:
                continue

            # Build Graph structures locally (CPU)
            edges_arr = np.array(edges_list)
            src, r, dst = edges_arr[:, 0], edges_arr[:, 1], edges_arr[:, 2]
            node_ids_sub = np.unique(np.concatenate([src, dst]))
            node_to_idx = {old: idx for idx, old in enumerate(node_ids_sub)}
            
            g = dgl.graph(([node_to_idx[s] for s in src], [node_to_idx[d] for d in dst]), num_nodes=len(node_ids_sub))
            
            dgl_graphs.append(g)
            all_node_ids.append(torch.LongTensor(node_ids_sub))
            all_etypes.append(torch.LongTensor(r))
            node_mappings[i] = node_to_idx
        
        # --- STEP 2: SINGLE BATCHED GAT PASS (THE SPEEDUP) ---
        gat_embs_per_graph = {}
        if len(dgl_graphs) > 0:
            # Fuse all small graphs into one massive disjoint graph and move to GPU
            batched_g = dgl.batch(dgl_graphs).to(device)
            batched_node_ids = torch.cat(all_node_ids, dim=0).to(device)
            batched_etypes = torch.cat(all_etypes, dim=0).to(device)
            
            # Execute GAT EXACTLY ONCE for the entire batch
            base_emb_batched = self.ent_embeddings(batched_node_ids)
            gat_emb_batched = self.gat(batched_g, base_emb_batched, self.rel_embeddings.weight, batched_etypes)
            
            # Split the massive output tensor back into individual graph chunks
            split_sizes = batched_g.batch_num_nodes().tolist()
            split_embs = torch.split(gat_emb_batched, split_sizes)
            
            valid_idx = 0
            for i in range(batch_size):
                if i in node_mappings:
                    gat_embs_per_graph[i] = split_embs[valid_idx]
                    valid_idx += 1

        # --- STEP 3: EXTRACT FEATURES ---
        all_q_id_embs = []
        all_q_ctx_embs = []
        all_e_id_embs = []
        all_e_ctx_embs = []
        for i in range(batch_size):
            q_id_emb = self.ent_embeddings(query_ids[i].unsqueeze(0))
            e_id_embs_batch = self.ent_embeddings(entity_ids[i])
            
            all_q_id_embs.append(q_id_emb)
            all_e_id_embs.append(e_id_embs_batch)
            
            if i not in node_mappings:
                # Fallback for subgraphs that were too small
                all_q_ctx_embs.append(q_id_emb)
                all_e_ctx_embs.append(e_id_embs_batch)
            else:
                # Extract specific node features from the split GAT results
                node_to_idx = node_mappings[i]
                gat_emb_sub = gat_embs_per_graph[i]
                
                qid = query_ids[i].item()
                q_ctx_emb = gat_emb_sub[node_to_idx[qid]].unsqueeze(0) if qid in node_to_idx else q_id_emb
                all_q_ctx_embs.append(q_ctx_emb)

                eids = entity_ids[i].tolist()
                in_graph = [eid for eid in eids if eid in node_to_idx]
                out_graph = [eid for eid in eids if eid not in node_to_idx]
 
                # Pre-fetch all out-of-graph embeddings in one shot
                if out_graph:
                    missing_ids = torch.LongTensor(out_graph).to(device)
                    missing_embs = self.ent_embeddings(missing_ids)           # (M, D)
                    missing_map = {eid: missing_embs[j] for j, eid in enumerate(out_graph)}
                else:
                    missing_map = {}

                
                e_ctx_list = []
                for eid in eids:
                    if eid in node_to_idx:
                        e_ctx_list.append(gat_emb_sub[node_to_idx[eid]].unsqueeze(0))
                    else:
                        e_ctx_list.append(missing_map[eid].unsqueeze(0))
                all_e_ctx_embs.append(torch.cat(e_ctx_list, dim=0))


        # 3. Final Batched Adapter Calls (Resolves BatchNorm Crash)
        q_id_final = torch.cat(all_q_id_embs, dim=0)
        q_ctx_final = torch.cat(all_q_ctx_embs, dim=0)
        query_output = self.adapter(q_id_final, q_ctx_final) # (Batch, 4096)

        e_id_final = torch.cat(all_e_id_embs, dim=0)
        e_ctx_final = torch.cat(all_e_ctx_embs, dim=0)
        entity_output = self.adapter(e_id_final, e_ctx_final) # (Batch * K, 4096)

        return query_output, entity_output


def build_rerank_enhancer(ent_path, rel_path, graph_weights_path, llm_hidden_size,
                           adapter_size=512, gnn_hidden_dim=128, gnn_layers=1,
                           load_stage1=True):
    """Build the Stage-2 (LLM) BioEGAT enhancer — Option A: reuse Stage-1 graph, replace projection.

    The KBGAT message-passing + InteractE conv/fc1 feature extractor and the fine-tuned
    entity/relation embeddings are loaded from the Stage-1 reranker checkpoint
    (``graph_only_best_{ds}.pt``) and FROZEN. The InteractE adapter's final projection
    ``adapter.fc2`` is re-initialised from ``adapter_size -> llm_hidden_size`` (the LLM token
    space) and is the ONLY trainable graph component (trained jointly with LoRA).

    Returns ``(enhancer, info)`` where ``info`` reports the derived dims.

    Set ``load_stage1=False`` for inference: build the identical architecture, then load the
    Stage-2 ``graph_model.bin`` over it (which already carries the trained fc2).
    """
    kge_ent = torch.load(ent_path, map_location="cpu").float()
    kge_rel = torch.load(rel_path, map_location="cpu").float()
    emb_dim = kge_ent.shape[1]

    # Reconstruct the Stage-1 enhancer EXACTLY (output_size = emb_dim) so rerank weights load 1:1.
    enhancer = GraphEnhancer(
        kge_embedding=kge_ent, input_size=emb_dim, rel_input=kge_rel,
        gnn_hidden_dim=gnn_hidden_dim, gnn_num_hidden_layers=gnn_layers,
        adapter_size=adapter_size, output_size=emb_dim, hidden_act="silu", freeze_ent=True,
    )

    if load_stage1:
        state = torch.load(graph_weights_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        # GraphReranker saved its enhancer under the "enhancer." prefix — strip it.
        enh_state = {k[len("enhancer."):]: v for k, v in state.items() if k.startswith("enhancer.")}
        if not enh_state:  # already a bare enhancer state_dict
            enh_state = state
        missing, unexpected = enhancer.load_state_dict(enh_state, strict=False)
        if unexpected:
            print(f"[BioEGAT] load_state_dict unexpected keys: {unexpected}")
        # only adapter.fc2/bn2 (replaced below) and nothing else should be missing
        crit = [m for m in missing if not (m.startswith("adapter.fc2") or m.startswith("adapter.bn2"))]
        assert not crit, f"[BioEGAT] missing critical Stage-1 keys: {crit}"

    # --- Replace the projection: adapter.fc2 (adapter_size -> emb_dim) -> (adapter_size -> llm_hidden) ---
    enhancer.adapter.fc2 = nn.Linear(adapter_size, llm_hidden_size)
    nn.init.xavier_uniform_(enhancer.adapter.fc2.weight)
    nn.init.zeros_(enhancer.adapter.fc2.bias)

    # Freeze the whole backbone, then unfreeze only the new projection head.
    for p in enhancer.parameters():
        p.requires_grad = False
    for p in enhancer.adapter.fc2.parameters():
        p.requires_grad = True

    info = {"emb_dim": emb_dim, "num_relations": int(kge_rel.shape[0]),
            "rel_dim": int(kge_rel.shape[1]), "adapter_size": adapter_size}
    return enhancer, info
