from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer, SlimFC
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
import Database
import dgl
from dgl.nn.pytorch.conv import GATv2Conv

torch, nn = try_import_torch()


def _p(x, name):
    if isinstance(x, torch.Tensor):
        print(f"[DBG] {name:22s} shape={tuple(x.shape)} dtype={x.dtype}")
    elif isinstance(x, dgl.DGLGraph):
        print(f"[DBG] {name:22s} nodes={x.num_nodes()} edges={x.num_edges()}")
    else:
        try:
            print(f"[DBG] {name:22s} {x}")
        except:
            print(f"[DBG] {name:22s} (non-printable)")


class AuGraphModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        self.original_space = obs_space.original_space

        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        u, v = Database.u, Database.v
        self.g = dgl.graph((u, v))
        self.lg = dgl.line_graph(self.g, backtracking=False)

        num_edges = self.original_space['phylink'].shape[0]
        in_feats = self.original_space['phylink'].shape[1]
        n_hidden = 32
        num_heads = 3

        # ---------- GNN ----------
        self._edge_gnn = nn.ModuleList()
        self._edge_gnn.append(GATv2Conv(in_feats, n_hidden, num_heads=num_heads, allow_zero_in_degree=True))
        self._edge_gnn.append(GATv2Conv(n_hidden*num_heads, n_hidden, num_heads=num_heads, allow_zero_in_degree=True))

        gnn_flat_size = num_edges * num_heads * n_hidden

        gnn_fc_stack_config = {
            "fcnet_hiddens": model_config.get("fcnet_hiddens"),
            "fcnet_activation": model_config.get("fcnet_activation"),
        }

        gnn_fc_layers = []
        in_size = gnn_flat_size
        for i, out_size in enumerate(gnn_fc_stack_config['fcnet_hiddens']):
            gnn_fc_layers.append(
                SlimFC(
                    in_size=in_size,
                    out_size=out_size,
                    activation_fn=gnn_fc_stack_config["fcnet_activation"],
                    initializer=normc_initializer(1.0)
                )
            )
            in_size = out_size
        self.gnn_hidden = nn.Sequential(*gnn_fc_layers)  # 全连接层输出

        concat_size = gnn_fc_stack_config['fcnet_hiddens'][-1]  # 记录处理后连接起来的总长
        concat_size += (self.original_space['request_src'].high - self.original_space['request_src'].low)[0] + 1
        concat_size += (self.original_space['request_dest'].high - self.original_space['request_dest'].low)[0] + 1
        concat_size += self.original_space['request_traffic'].shape[0]

        post_fc_stack_config = {
            "fcnet_hiddens": model_config.get("post_fcnet_hiddens"),
            "fcnet_activation": model_config.get("post_fcnet_activation"),
        }

        post_fc_layers = []
        in_size = concat_size   # FC输入层神经元个数，
        for i, out_size in enumerate(post_fc_stack_config['fcnet_hiddens']):
            post_fc_layers.append(
                SlimFC(
                    in_size=in_size,
                    out_size=out_size,
                    activation_fn=post_fc_stack_config["fcnet_activation"],
                    initializer=normc_initializer(1.0)
                )
            )
            in_size = out_size
        self._hidden = nn.Sequential(*post_fc_layers)   # 全连接层输出

        # Actions and value heads.
        self.logits_layer = None
        self.value_layer = None
        self._value_out = None
        # if num_outputs:
        #     self.logits_layer = SlimFC(
        #         in_size=out_size,
        #         out_size=num_outputs,
        #         activation_fn=None,
        #     )
        #     # Create the value branch model.
        #     self.value_layer = SlimFC(
        #         in_size=out_size,
        #         out_size=1,
        #         activation_fn=None,
        #         initializer=normc_initializer(0.01))  # 正态分布初始化

    @override(ModelV2)  # 子类覆写父类函数
    def forward(self, input_dict, state, seq_lens):
        phylink = input_dict["obs"]["phylink"]
        request_src = input_dict["obs"]["request_src"]
        request_dest = input_dict["obs"]["request_dest"]
        request_traffic = input_dict["obs"]["request_traffic"]

        B, E, F = phylink.shape         # B个样本，每个样本E条物理链路，每条链路F维特征
        blg = dgl.batch([self.lg] * B)  # batch化的线图：B个完全相同的 line_graph
        h = phylink.reshape(B * E, F)   # 把(B,E,F)摊平成(B*E,F)，作为 line_graph 上每个“节点”的特征

        for li, layer in enumerate(self._edge_gnn):
            h_out = layer(blg, h)  # GAT: 内部已经做了消息传递
            h = h_out.reshape(h_out.shape[0], -1)  # (B*E, num_heads * n_hidden) = (B*E, 96)

        gat_out = h.reshape(B, E, -1)  # (B, E, 96)
        # x_gnn = gat_out.mean(dim=1)  # (B, 96)
        x_gnn = gat_out.reshape(B, -1)  # (B, 24*96=2304)
        x_gnn = self.gnn_hidden(x_gnn)

        outs = []
        outs.append(x_gnn)

        out_onehot_src = nn.functional.one_hot(request_src.long(), (self.original_space['request_src'].high - self.original_space['request_src'].low)[0] + 1)
        out_onehot_src = out_onehot_src.squeeze(dim=1).float()  # 去除空维度
        outs.append(out_onehot_src)

        out_onehot_dest = nn.functional.one_hot(request_dest.long(), (self.original_space['request_dest'].high - self.original_space['request_dest'].low)[0] + 1)
        out_onehot_dest = out_onehot_dest.squeeze(dim=1).float()  # 去除空维度
        outs.append(out_onehot_dest)

        outs.append(request_traffic)

        out_add = torch.cat(outs, dim=1)  # gnn_hidden[1]+9+9+24
        out = self._hidden(out_add)

        # if not self.logits_layer is None:
        #     logits, values = self.logits_layer(out), self.value_layer(out)
        #     # print(logits.shape)
        #     self._value_out = torch.reshape(values, [-1])  # 表示将矩阵形式的value展平
        #     # return logits + inf_mask, []
        #     return logits, []
        # else:
        #     return out, []
        return out, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out