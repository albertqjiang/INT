import torch
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.utils import remove_self_loops
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv  # noqa
from torch_geometric.nn.inits import reset
from torch_geometric.nn import GraphConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils import softmax
import torch.nn as nn
# import dgl
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(0)


class GINMeanConv(torch.nn.Module):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (nn.Sequential): Neural network :math:`h_{\mathbf{\Theta}}`.
        eps (float, optional): (Initial) :math:`\epsilon` value.
            (default: :obj:`0`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
    """

    def __init__(self, nn, eps=0, train_eps=False):
        super(GINMeanConv, self).__init__()
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        row, col = edge_index
        out = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))
        out = (1 + self.eps) * x + out
        out = self.nn(out)
        return out

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class GINConv(torch.nn.Module):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (nn.Sequential): Neural network :math:`h_{\mathbf{\Theta}}`.
        eps (float, optional): (Initial) :math:`\epsilon` value.
            (default: :obj:`0`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
    """

    def __init__(self, nn, state_dim=1024, eps=0, train_eps=False):
        super(GINConv, self).__init__()
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps).to(device)

    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        row, col = edge_index
        out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
        out = (1 + self.eps) * x + out
        out = self.nn(out)
        return out

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class CondGINConv(torch.nn.Module):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (nn.Sequential): Neural network :math:`h_{\mathbf{\Theta}}`.
        eps (float, optional): (Initial) :math:`\epsilon` value.
            (default: :obj:`0`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
    """

    def __init__(self, nn, state_dim, eps=0, negative_slope=0.2, train_eps=False):
        super(CondGINConv, self).__init__()
        self.nn = nn
        self.initial_eps = eps
        self.negative_slope = negative_slope
        self.state_dim = state_dim
        self.key_transform = torch.nn.Linear(256, 2 * state_dim, bias=False)
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, condition, ind=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        row, col = edge_index
        x_i = x[col]
        x_j = x[row]
        key = self.key_transform(condition)
        if ind is not None:
            batch_gnn_ind, gnn_ind = ind
            batch_gnn_ind = torch.LongTensor(batch_gnn_ind).to(device)
            gnn_ind = torch.LongTensor(gnn_ind).to(device)
            key = torch.index_select(key, 0, batch_gnn_ind)
            key = torch.index_select(key, 0, gnn_ind)
        alpha = (torch.cat([x_i, x_j], dim=-1) * key).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = torch.sigmoid(alpha).view(-1, 1)
        out = scatter_add(x[col]*alpha, row, dim=0, dim_size=x.size(0))

        out = (1 + self.eps) * x + out
        out = self.nn(out)
        return out

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class GraphEncoder(nn.Module):
    def __init__(self, num_in, num_out, conv_dim=64, hidden_layers=1, norm=None):
        super(GraphEncoder, self).__init__()
        self.hidden_layers = hidden_layers
        self.conv_in = GCNConv(num_in, conv_dim, cached=False)
        self.hidden_convs = nn.ModuleList([
            GCNConv(conv_dim, conv_dim, cached=False) for _ in range(self.hidden_layers)
        ])
        self.conv_out = GCNConv(conv_dim, num_out, cached=False)
        self.norm = norm
        if norm is not None:
            if norm == "bn":
                self.norm_in = nn.BatchNorm1d(conv_dim)
                self.hidden_norms = nn.ModuleList([
                    nn.BatchNorm1d(conv_dim) for _ in range(self.hidden_layers)
                ])
            elif norm == "ln":
                self.norm_in = nn.LayerNorm(conv_dim)
                self.hidden_norms = nn.ModuleList([
                    nn.LayerNorm(conv_dim) for _ in range(self.hidden_layers)
                ])
            else:
                raise NotImplementedError
        self.to(device)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv_in(x, edge_index))
        if self.norm is not None:
            x = self.norm_in(x)
        for i in range(self.hidden_layers):
            x = F.relu(self.hidden_convs[i](x, edge_index))
            if self.norm is not None:
                x = self.hidden_norms[i](x)
        x = self.conv_out(x, edge_index)
        return x


class GraphTransformingEncoder(nn.Module):
    def __init__(self, num_in, num_out, conv_dim=64, heads=1, hidden_layers=1, norm=None):
        super(GraphTransformingEncoder, self).__init__()
        self.hidden_layers = hidden_layers
        self.conv_in = GATConv(num_in, conv_dim, heads=heads)
        self.hidden_convs = nn.ModuleList([
            GATConv(heads * conv_dim, conv_dim, heads=heads) for _ in range(self.hidden_layers)
        ])
        self.conv_out = GATConv(heads * conv_dim, int(num_out / heads), heads=heads)
        self.norm = norm
        if norm is not None:
            if norm == "bn":
                self.norm_in = nn.BatchNorm1d(conv_dim)
                self.hidden_norms = nn.ModuleList([
                    nn.BatchNorm1d(conv_dim) for _ in range(self.hidden_layers)
                ])
            elif norm == "ln":
                self.norm_in = nn.LayerNorm(conv_dim)
                self.hidden_norms = nn.ModuleList([
                    nn.LayerNorm(conv_dim) for _ in range(self.hidden_layers)
                ])
            else:
                raise NotImplementedError
        self.to(device)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv_in(x, edge_index))
        if self.norm is not None:
            x = self.norm_in(x)
        for i in range(self.hidden_layers):
            x = F.relu(self.hidden_convs[i](x, edge_index))
            if self.norm is not None:
                x = self.hidden_norms[i](x)
        x = self.conv_out(x, edge_index)
        return x


class RawGATEncoder(nn.Module):
    def __init__(self, num_in, num_out, conv_dim=64, heads=1, hidden_layers=1, norm=None):
        super(RawGATEncoder, self).__init__()
        self.hidden_layers = hidden_layers
        self.conv_in = GATConv(num_in, conv_dim, heads=heads)
        self.hidden_convs = nn.ModuleList([
            GATConv(heads * conv_dim, conv_dim, heads=heads) for _ in range(self.hidden_layers)
        ])
        self.conv_out = GATConv(heads * conv_dim, int(num_out / heads), heads=heads)
        self.to(device)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv_in(x, edge_index))
        for i in range(self.hidden_layers):
            x = F.relu(self.hidden_convs[i](x, edge_index))
        x = self.conv_out(x, edge_index)
        return x


class TransGATEncoder(torch.nn.Module):
    def __init__(self, input_dim, inception=5, hidden_dim=64, heads=8, gat_dropout_rate=0.1, dropout_rate=0.1):
        super(TransGATEncoder, self).__init__()
        self.inception = inception
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dim_per_head = int(hidden_dim / heads)
        self.gat_dropout_rate = gat_dropout_rate
        self.dropout_rate = dropout_rate

        self.conv1 = GATConv(input_dim, self.dim_per_head, heads=self.heads, dropout=self.gat_dropout_rate)
        self.convs = nn.ModuleList(
            [GATConv(self.hidden_dim, self.dim_per_head, heads=self.heads, dropout=self.gat_dropout_rate) for _ in
             range(self.inception)])
        self.conv_out = GATConv(self.hidden_dim, self.hidden_dim, heads=1, dropout=self.gat_dropout_rate, concat=False)
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.inception)])
        self.fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Dropout(p=self.dropout_rate),
            )

            for _ in range(self.inception)])
        self.fc_norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.inception)])
        self.to(device)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        for i in range(self.inception):
            x1 = self.convs[i](x, edge_index)
            x1 = F.elu(x1)
            x1 = F.dropout(x1, p=self.dropout_rate, training=self.training)
            x = self.norms[i](x + x1)
            x1 = self.fcs[i](x)
            x = self.fc_norms[i](x + x1)
        return x


class FCResBlock(nn.Module):
    def __init__(self, nn_dim, use_layer_norm=True):
        self.use_layer_norm = use_layer_norm
        super(FCResBlock, self).__init__()
        self.norm_in = nn.LayerNorm(nn_dim)
        self.norm_out = nn.LayerNorm(nn_dim)
        self.transform1 = torch.nn.Linear(nn_dim, nn_dim)
        torch.nn.init.normal_(self.transform1.weight, std=0.005)
        self.transform2 = torch.nn.Linear(nn_dim, nn_dim)
        torch.nn.init.normal_(self.transform2.weight, std=0.005)

    def forward(self, x):
        if self.use_layer_norm:
            x_branch = self.norm_in(x)
        else:
            x_branch = x
        x_branch = self.transform1(F.relu(x_branch))
        if self.use_layer_norm:
            x_branch = self.norm_out(x_branch)
        x_out = x + self.transform2(F.relu(x_branch))
        return x_out


class MLPBlock(nn.Module):
    def __init__(self, nn_dim, use_layer_norm=False, ):
        self.use_layer_norm = use_layer_norm
        super(MLPBlock, self).__init__()
        self.norm_in = nn.LayerNorm(nn_dim)
        self.norm_out = nn.LayerNorm(nn_dim)
        self.transform1 = torch.nn.Linear(nn_dim, nn_dim)
        torch.nn.init.normal_(self.transform1.weight, std=0.005)
        self.transform2 = torch.nn.Linear(nn_dim, nn_dim)
        torch.nn.init.normal_(self.transform2.weight, std=0.005)

    def forward(self, x):
        if self.use_layer_norm:
            x_branch = self.norm_in(x)
        else:
            x_branch = x
        if self.use_layer_norm:
            x_branch = self.norm_out(x_branch)
        x_out = self.transform2(F.relu(x_branch))
        return x_out


class GraphIsomorphismEncoder(nn.Module):
    def __init__(self, num_in, num_out, nn_dim=64, hidden_layers=1, norm=None):
        super(GraphIsomorphismEncoder, self).__init__()
        gin_nn_in = nn.Sequential(
            nn.Linear(num_in, nn_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nn_dim, nn_dim, bias=False)
        )
        gin_nn_in.to(device)
        ginconv = GINConv
        self.gin_conv_in = ginconv(gin_nn_in, state_dim=num_in)

        self.hidden_layers = hidden_layers
        # self.hidden_nns = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(nn_dim, nn_dim, bias=False),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(nn_dim, nn_dim, bias=False)
        #     ) for _ in range(self.hidden_layers)
        # ])
        self.hidden_nns = nn.ModuleList([
            MLPBlock(nn_dim) for _ in range(self.hidden_layers)
        ])

        self.hidden_gins = nn.ModuleList([
            ginconv(self.hidden_nns[i], state_dim=nn_dim) for i in range(self.hidden_layers)
        ])

        # gin_nn_out = nn.Sequential(
        #     nn.Linear(nn_dim, nn_dim, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(nn_dim, num_out, bias=False)
        # )
        # gin_nn_out.to(device)
        # self.gin_conv_out = ginconv(gin_nn_out, state_dim=nn_dim)
        self.gin_conv_out = ginconv(MLPBlock(nn_dim), state_dim=nn_dim)
        self.norm = None
        # if norm is not None:
        #     if norm == "bn":
        #         self.norm_in = nn.BatchNorm1d(nn_dim)
        #         self.hidden_norms = nn.ModuleList([
        #             nn.BatchNorm1d(nn_dim) for _ in range(self.hidden_layers)
        #         ])
        #     elif norm == "ln":
        #         self.norm_in = nn.LayerNorm(nn_dim)
        #         self.hidden_norms = nn.ModuleList([
        #             nn.LayerNorm(nn_dim) for _ in range(self.hidden_layers)
        #         ])
        #     else:
        #         raise NotImplementedError
        # self.to(device)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gin_conv_in(x, edge_index))
        if self.norm is not None:
            x = self.norm_in(x)
        for i in range(self.hidden_layers):
            x = F.relu(self.hidden_gins[i](x, edge_index))
            if self.norm is not None:
                x = self.hidden_norms[i](x)
        x = self.gin_conv_out(x, edge_index)
        return x


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox['h']))
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}


class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        # concatenate h_jl for equation (1), (2), (3), (4)
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        # equation (2)
        f = torch.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
        # second term of equation (5)
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_cat), 'c': c}

    def apply_node_func(self, nodes):
        # equation (1), (3), (4)
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        # equation (5)
        c = i * u + nodes.data['c']
        # equation (6)
        h = o * torch.tanh(c)
        return {'h' : h, 'c' : c}


class TreeLSTM(nn.Module):
    def __init__(self,
                 num_in,
                 num_out,
                 h_size,
                 dropout=0.5):
        super(TreeLSTM, self).__init__()
        self.embedding = nn.Linear(num_in, h_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_out)
        self.cell = ChildSumTreeLSTMCell(h_size, h_size)

    def forward(self, batch, h, c):
        """Compute tree-lstm prediction given a batch.

        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.

        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        g = batch
        g.register_message_func(self.cell.message_func)
        g.register_reduce_func(self.cell.reduce_func)
        g.register_apply_node_func(self.cell.apply_node_func)
        # feed embedding
        embeds = self.embedding(g.ndata["feat"])
        g.ndata['iou'] = self.cell.W_iou(self.dropout(embeds))
        g.ndata['h'] = h
        g.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(g)
        # compute logits
        h = self.dropout(g.ndata.pop('h'))
        logits = self.linear(h)
        return logits




class SAGPooling(torch.nn.Module):
    r"""The self-attention pooling operator from the `"Self-Attention Graph
    Pooling" <https://arxiv.org/abs/1904.08082>`_ and `"Understanding
    Attention and Generalization in Graph Neural Networks"
    <https://arxiv.org/abs/1905.02850>`_ papers

    if :obj:`min_score` :math:`\tilde{\alpha}` is :obj:`None`:

        .. math::
            \mathbf{y} &= \textrm{GNN}(\mathbf{X}, \mathbf{A})

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot
            \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

    if :obj:`min_score` :math:`\tilde{\alpha}` is a value in [0, 1]:

        .. math::
            \mathbf{y} &= \mathrm{softmax}(\textrm{GNN}(\mathbf{X},\mathbf{A}))

            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot \mathbf{y})_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},

    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`.
    Projections scores are learned based on a graph neural network layer.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
            This value is ignored if min_score is not None.
            (default: :obj:`0.5`)
        GNN (torch.nn.Module, optional): A graph neural network layer for
            calculating projection scores (one of
            :class:`torch_geometric.nn.conv.GraphConv`,
            :class:`torch_geometric.nn.conv.GCNConv`,
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.SAGEConv`). (default:
            :class:`torch_geometric.nn.conv.GraphConv`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:`min_score` is used. (default: :obj:`1`)
        nonlinearity (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.tanh`)
        **kwargs (optional): Additional parameters for initializing the graph
            neural network layer.
    """

    def __init__(self, in_channels, ratio=0.5, GNN=GraphConv, min_score=None,
                 multiplier=1, nonlinearity=torch.tanh, **kwargs):
        super(SAGPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.gnn = GNN(in_channels, 1, **kwargs)
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()


    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = self.gnn(attn, edge_index).view(-1)

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]


    def __repr__(self):
        return '{}({}, {}, {}={}, multiplier={})'.format(
            self.__class__.__name__, self.gnn.__class__.__name__,
            self.in_channels,
            'ratio' if self.min_score is None else 'min_score',
            self.ratio if self.min_score is None else self.min_score,
            self.multiplier)