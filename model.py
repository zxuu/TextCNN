import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, args):
        """参数

        Args:
            args (_type_): parser.parse_args()
        """
        super(TextCNN, self).__init__()
        self.args = args

        class_num = args.class_num    # 几分类
        chanel_num = 1
        filter_num = args.filter_num    # default 100
        filter_sizes = args.filter_sizes    # default [3,4,5]

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)    # 初始化字典向量
        if args.static:    # false
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        if args.multichannel:    # false
            self.embedding2 = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(args.vectors)
            chanel_num += 1
        else:
            self.embedding2 = None
        self.convs = nn.ModuleList(
            # 三个卷积层 这里拿三个卷积进行卷积
            # nn.Conv2d(1,100,(3,128))
            # nn.Conv2d(1,100,(4,128))
            # nn.Conv2d(1,100,(5,128))
            [nn.Conv2d(chanel_num, filter_num, (size, embedding_dimension)) for size in filter_sizes]
            )
        self.dropout = nn.Dropout(args.dropout)    # 0.5
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)

    def forward(self, x):
        """前向传播

        Args:
            x (tensor): [batch, max(seq_len)]

        Returns:
            tensor: [batch, 3]
        """
        if self.embedding2:    # false
            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1)
        else:
            x = self.embedding(x)    # [batch, max(seq_len), dim]
            x = x.unsqueeze(1)    # [batch, 1, max(seq_len), dim]
        # conv()之后 [batch, filter_num, seq_len-size+1, 1]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]    # [[batch, filter_num, seq_len-size+1], [...], [...]]
        # 最大池化
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]    # [[batch, filter_num], [], []]
        # 拼接3个卷积的结果
        x = torch.cat(x, 1)    # [batch, 3*filter_num]
        x = self.dropout(x)
        logits = self.fc(x)
        # 得到logits之后进行交叉熵计算 cross_entropy(logits, label)
        return logits
