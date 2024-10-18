import settings
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F



class LSTMCAE(nn.Module):
    """
    LSTM-based Convolutional Autoencoder module for C-MAPSS dataset.
    Based on paper of comparitive study
    """
    def __init__(
        self, n_inputs, length):
        super().__init__()

        # Building an encoder
        self.lstm_enc = nn.LSTM(
            input_size=n_inputs,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.act_fun = nn.Tanh()

        self.cnn1d_enc = nn.Conv1d(32, 16, kernel_size=2, stride=2)

        self.cnn1d_dec = nn.ConvTranspose1d(16, 32, kernel_size=2, stride=2)
        self.lstm_dec = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.fc = nn.Linear(32, n_inputs)



        self.length = length
        self.n_inputs = n_inputs

    def forward(self, time_series):
        x_res_enc, _ = self.lstm_enc(time_series)
        #x = self.act_fun(x)
        # change the dimensions from [batch, seq, features] to [batch, features, seq]
        x = x_res_enc.permute(0, 2, 1)
        x = self.cnn1d_enc(x)
        x = self.cnn1d_dec(x)
        x = x.permute(0, 2, 1)
        x_res_dec, _ = self.lstm_dec(x)
        x_res = x_res_enc + x_res_dec
        x = self.fc(x_res)
        return x



class Monotonic_Layer(nn.Linear):
    """
    Monotonic module as described in the paper. The acttivation function should always be monotonic.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Monotonic_Layer, self).__init__(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.seed(settings.seed_number)
        self.reset_parameters()

    def seed(self, seed_number=None):
        torch.manual_seed(seed_number)
        torch.cuda.manual_seed(seed_number)
        torch.cuda.manual_seed_all(seed_number)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input, first_layer=False):
        if first_layer:
            # apply a linear layer to the first n-1 features and the corresponding self.weight dimension
            out1 = F.linear(input[:, :-1], self.weight[:, :-1], self.bias)
            out2 = F.linear(
                input[:, -1].view(-1, 1),
                torch.exp(self.weight[:, -1].view(-1, 1)),
                None,
            )
            out = torch.add(out1, out2)
        else:
            out = F.linear(input, torch.exp(self.weight), self.bias)
        return out


class PositiveBatchNorm1d(nn.BatchNorm1d):
    """
    BatchNorm1d with positive weights. This is used to ensure that the monotonicity constraint is satisfied.
    """

    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(PositiveBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input):
        self._check_input_dim(input)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / self.num_batches_tracked.float()
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        out = F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            torch.exp(self.weight),
            self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )
        return out


class HalfPosTanh(nn.Module):
    def __init__(self):
        super(HalfPosTanh, self).__init__()

    def forward(self, x):
        x = (F.tanh(x) + 1) / 2
        return x


class AE(torch.nn.Module):
    """
    Autoencoder module with monotonicity constraint. This is used for pretraining the encoder.
    """

    def __init__(
        self, n_inputs, length, n_features, hidden_dim, dr_rate=0.1, use_demo=False
    ):
        super().__init__()

        self.seed(settings.seed_number)

        self.hidden_dim = hidden_dim
        if use_demo:
            add_features = 18
        else:
            add_features = 1

        # Building an encoder
        self.lstm_enc = nn.LSTM(
            input_size=n_inputs,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )
        self.act_fun = nn.Tanh()
        self.flatten = nn.Flatten()
        self.first_layer = nn.Linear(self.hidden_dim * length, self.hidden_dim)
        self.act_fun2 = nn.Softplus()
        self.dropout = nn.Dropout(dr_rate)
        self.batch_norm_first = nn.BatchNorm1d(self.hidden_dim)

        self.first_monotonic_layer = Monotonic_Layer(self.hidden_dim + add_features, 32)
        self.encoder = nn.Sequential(
            self.act_fun2,
            PositiveBatchNorm1d(32, affine=True),
            Monotonic_Layer(32, 16),
            self.act_fun2,
            PositiveBatchNorm1d(16, affine=True),
            Monotonic_Layer(16, n_features),
        )

        # Building a decoder
        self.decoder = nn.Sequential(
            Monotonic_Layer(n_features, 16),
            self.act_fun2,
            PositiveBatchNorm1d(16, affine=True),
            Monotonic_Layer(16, 32),
            self.act_fun2,
            PositiveBatchNorm1d(32, affine=True),
            Monotonic_Layer(32, self.hidden_dim + add_features),
            self.act_fun2,
            nn.BatchNorm1d(self.hidden_dim + add_features),
        )
        self.last_layer = nn.Linear(
            self.hidden_dim + add_features - 1, self.hidden_dim * length
        )
        self.act_fun3 = nn.Tanh()
        self.dropout = nn.Dropout(dr_rate)

        self.lstm_dec = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=n_inputs,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )
        self.length = length
        self.n_inputs = n_inputs
        self.use_demo = use_demo
        self.n_features = n_features

    def seed(self, seed_number=None):
        torch.manual_seed(seed_number)
        torch.cuda.manual_seed(seed_number)
        torch.cuda.manual_seed_all(seed_number)

    def disable_bn(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """

        for layer in [self.batch_norm_first, self.dropout, self.encoder, self.decoder]:
            for m in layer.modules():
                if isinstance(m, nn.BatchNorm1d):
                    if mode:
                        m.eval()
                    else:
                        m.train()
                        m.weight.requires_grad = True
                        m.bias.requires_grad = True
                elif isinstance(m, nn.Dropout):
                    if mode:
                        m.eval()
                    else:
                        m.train()

    def forward(self, time_series, time, demo):
        x, _ = self.lstm_enc(time_series)
        x = self.act_fun(x)
        x = self.flatten(x)
        encoded_hidden_dim = self.dropout(self.act_fun2(self.first_layer(x)))
        encoded_hidden_dim = self.batch_norm_first(encoded_hidden_dim)
        if demo is not None:
            encoded_hidden_dim = torch.cat((encoded_hidden_dim, demo), dim=1)
        x = torch.cat((encoded_hidden_dim, time.reshape(-1, 1)), dim=1)
        x = self.first_monotonic_layer(x, first_layer=True)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded, reconstructed_time = decoded[:, :-1], decoded[:, -1]

        x = self.dropout(self.act_fun3(self.last_layer(decoded)))
        x = x.reshape(-1, self.length, self.hidden_dim)
        x, _ = self.lstm_dec(x)

        return x, reconstructed_time, encoded, encoded_hidden_dim, decoded


class AE_ACOUSTIC_DIC(torch.nn.Module):
    """
    Autoencoder with monotonic layers
    """

    def __init__(
        self,
        n_inputs_acoustic,
        n_inputs_dic,
        length_acoustic,
        length_dic,
        height,
        width,
        n_features,
        hidden_dim,
        dr_rate=0.2,
    ):
        super(AE_ACOUSTIC_DIC, self).__init__()
        self.seed(settings.seed_number)

        self.n_inputs_acoustic = n_inputs_acoustic
        self.n_inputs_dic = n_inputs_dic
        self.L_acoustic = length_acoustic
        self.L_dic = length_dic
        self.H = height
        self.W = width
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # Encoder layers

        # Encoder - DIC part

        self.cnn3d_enc = nn.Sequential(
            nn.Conv3d(
                n_inputs_dic, hidden_dim, kernel_size=3, stride=(1, 2, 2), padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_dim),
            nn.Dropout3d(p=dr_rate),
            nn.Conv3d(
                hidden_dim,
                2 * hidden_dim,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm3d(2 * hidden_dim),
            nn.Dropout3d(p=dr_rate),
            nn.Conv3d(
                2 * hidden_dim,
                2 * hidden_dim,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm3d(2 * hidden_dim),
            nn.Dropout3d(p=dr_rate),
            nn.Conv3d(
                2 * hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            ),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        len_dim, dim_shape = self.calculate_dims(layer="encoder")
        # print("encoder", len_dim, dim_shape)
        self.act_fun_soft = nn.Softplus()

        self.first_layer_dic = nn.Sequential(
            nn.Linear(len_dim, 300),
            self.act_fun_soft,
            nn.BatchNorm1d(300),
            nn.Linear(300, self.hidden_dim),
        )
        self.dropout = nn.Dropout(dr_rate)
        self.batch_norm_first_dic = nn.BatchNorm1d(self.hidden_dim)

        # Encoder - Acoustic part

        self.lstm_enc = nn.LSTM(
            input_size=n_inputs_acoustic,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )
        self.act_fun_tan = nn.Tanh()
        self.flatten = nn.Flatten()

        self.first_layer_acoustic = nn.Linear(
            self.hidden_dim * length_acoustic, self.hidden_dim
        )
        self.act_fun_tan_pos = HalfPosTanh()
        self.dropout = nn.Dropout(dr_rate)
        self.batch_norm_first_acoustic = nn.BatchNorm1d(self.hidden_dim)

        # Fusion part - Soft monotonic layers

        self.first_layer = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.batch_norm_first = nn.BatchNorm1d(self.hidden_dim)

        self.first_monotonic_layer = Monotonic_Layer(self.hidden_dim + 1, 32)
        self.encoder = nn.Sequential(
            self.act_fun_soft,
            PositiveBatchNorm1d(32, affine=True),
            Monotonic_Layer(32, 16),
            self.act_fun_soft,
            PositiveBatchNorm1d(16, affine=True),
            Monotonic_Layer(16, n_features),
        )

        # Defusion part - Soft monotonic layers

        self.decoder = nn.Sequential(
            Monotonic_Layer(n_features, 16),
            self.act_fun_soft,
            PositiveBatchNorm1d(16, affine=True),
            Monotonic_Layer(16, 32),
            self.act_fun_soft,
            PositiveBatchNorm1d(32, affine=True),
            Monotonic_Layer(32, self.hidden_dim + 1),
            self.act_fun_tan_pos,
            nn.BatchNorm1d(self.hidden_dim + 1),
        )
        self.last_layer = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)

        # Decoder - DIC part

        self.last_layer_dic = nn.Sequential(
            nn.Linear(self.hidden_dim, 300),
            self.act_fun_soft,
            nn.BatchNorm1d(300),
            nn.Linear(300, len_dim),
            self.act_fun_soft,
            nn.BatchNorm1d(len_dim),
            nn.Dropout(dr_rate),
        )

        self.cnn3d_dec = nn.Sequential(
            nn.Unflatten(1, (dim_shape[1], dim_shape[2], dim_shape[3], dim_shape[4])),
            nn.ConvTranspose3d(
                hidden_dim,
                2 * hidden_dim,
                kernel_size=2,
                stride=(1, 2, 2),
                padding=(1, 0, 0),
            ),
            nn.ReLU(),
            nn.BatchNorm3d(2 * hidden_dim),
            nn.Dropout3d(p=dr_rate),
            nn.ConvTranspose3d(
                2 * hidden_dim,
                2 * hidden_dim,
                kernel_size=2,
                stride=(1, 2, 2),
                padding=(0, 0, 0),
            ),
            nn.ReLU(),
            nn.BatchNorm3d(2 * hidden_dim),
            nn.Dropout3d(p=dr_rate),
            nn.ConvTranspose3d(
                2 * hidden_dim,
                hidden_dim,
                kernel_size=2,
                stride=(1, 2, 2),
                padding=(1, 0, 0),
            ),
            nn.ReLU(),
            nn.BatchNorm3d(hidden_dim),
            nn.Dropout3d(p=dr_rate),
            nn.ConvTranspose3d(
                hidden_dim,
                n_inputs_dic,
                kernel_size=2,
                stride=(1, 2, 2),
                padding=(0, 0, 0),
            ),
            nn.Sigmoid(),
        )
        # Decoder - Acoustic part

        self.last_layer_acoustic = nn.Linear(
            self.hidden_dim, self.hidden_dim * self.L_acoustic
        )

        self.dropout = nn.Dropout(dr_rate)

        self.lstm_dec = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=n_inputs_acoustic,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )

        # print("decoder", self.calculate_dims(layer='decoder'))

    def seed(self, seed_number=None):
        torch.manual_seed(seed_number)
        torch.cuda.manual_seed(seed_number)
        torch.cuda.manual_seed_all(seed_number)

    def disable_bn(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """

        for layer in [self.batch_norm_first, self.dropout, self.encoder, self.decoder]:
            for m in layer.modules():
                if isinstance(m, nn.BatchNorm1d):
                    if mode:
                        m.eval()
                    else:
                        m.train()
                        m.weight.requires_grad = True
                        m.bias.requires_grad = True
                elif isinstance(m, nn.Dropout):
                    if mode:
                        m.eval()
                    else:
                        m.train()

    def forward(self, time_series, time, demo):
        time_series, images = time_series

        # this is the DIC part

        x = self.cnn3d_enc(images)
        # print("1 layer", x[:5])
        x = self.act_fun_soft(x)
        x = self.flatten(x)
        # x = self.batch_norm0(x)

        encoded_hidden_dim_dic = self.first_layer_dic(x)
        encoded_hidden_dim_dic = self.dropout(self.act_fun_soft(encoded_hidden_dim_dic))
        encoded_hidden_dim_dic = self.batch_norm_first_dic(encoded_hidden_dim_dic)

        # this is the Acoustic part

        x, _ = self.lstm_enc(time_series)
        x = self.act_fun_tan(x)
        x = self.flatten(x)

        encoded_hidden_dim_acoustic = self.first_layer_acoustic(x)
        encoded_hidden_dim_acoustic = self.dropout(
            self.act_fun_soft(encoded_hidden_dim_acoustic)
        )
        encoded_hidden_dim_acoustic = self.batch_norm_first_acoustic(
            encoded_hidden_dim_acoustic
        )

        # this is the fusion part

        encoded_hidden_dim = torch.cat(
            (encoded_hidden_dim_acoustic, encoded_hidden_dim_dic), dim=1
        )
        encoded_hidden_dim = self.first_layer(encoded_hidden_dim)
        encoded_hidden_dim = self.dropout(self.act_fun_tan_pos(encoded_hidden_dim))
        encoded_hidden_dim = self.batch_norm_first(encoded_hidden_dim)
        if demo is not None:
            encoded_hidden_dim = torch.cat((encoded_hidden_dim, demo), dim=1)

        x = torch.cat((encoded_hidden_dim, time.reshape(-1, 1)), dim=1)

        x = self.first_monotonic_layer(x, first_layer=True)

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        # this is the defusion part

        decoded, reconstructed_time = decoded[:, :-1], decoded[:, -1]

        x = self.last_layer(decoded)

        decoded_acoustic = x[:, : self.hidden_dim]
        decoded_dic = x[:, self.hidden_dim :]

        # this is the DIC part

        decoded_dic = self.last_layer_dic(decoded_dic)
        decoded_dic = self.cnn3d_dec(decoded_dic)

        # this is the Acoustic part

        decoded_acoustic = self.last_layer_acoustic(decoded_acoustic)
        decoded_acoustic = self.dropout(self.act_fun_tan(decoded_acoustic))
        decoded_acoustic = decoded_acoustic.reshape(
            -1, self.L_acoustic, self.hidden_dim
        )
        decoded_acoustic, _ = self.lstm_dec(decoded_acoustic)
        decoded_acoustic = self.act_fun_tan(decoded_acoustic)

        x = (decoded_acoustic, decoded_dic)

        return x, reconstructed_time, encoded, encoded_hidden_dim, decoded

    def calculate_dims(
        self, layer, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        x = torch.rand(2, 1, self.L_dic, self.H, self.W)
        # print("input", x.shape)
        if layer == "encoder":
            return int(self.cnn3d_enc(x).numel() / 2), self.cnn3d_enc(x).shape


class ClusterAssignment(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        device: torch.device,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
    ):
        """
        Module to handle the soft assignment, where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.
        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterAssignment, self).__init__()

        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.seed(settings.seed_number)
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(
            initial_cluster_centers, requires_grad=True
        ).to(device)

    def seed(self, seed_number=None):
        torch.manual_seed(seed_number)
        torch.cuda.manual_seed(seed_number)
        torch.cuda.manual_seed_all(seed_number)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment (qij) for a batch of feature vectors, returning a batch of assignments
        for each cluster (see Equation 1 of the Methods section in the paper).
        :param x: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """

        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class DC(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        hidden_dimension: int,
        encoder: torch.nn.Module,
        device: torch.device,
        alpha_deg: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
    ):
        """
        Module for constructing the DSMC model, consisting of the pretrained encoder and the deep clustering (DC) model,
         this includes the pretrained encoder and the clustering process.
        :param cluster_number: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param encoder: encoder to use
        :param alpha_deg: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(DC, self).__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha_deg
        self.assignment = ClusterAssignment(
            cluster_number, self.hidden_dimension, device, alpha_deg, cluster_centers
        )

    def forward(
        self,
        time_series: torch.Tensor,
        time: torch.Tensor,
        demo: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.
        :param time_series: [batch size, length, features] FloatTensor
        :param time: [batch size, embedded_dimension] FloatTensor
        :param demo: Optional[[batch size, 2] FloatTensor], None]
        :return: [batch size, number of clusters] FloatTensor
        """

        (
            reconstructed,
            reconstructed_time,
            encoded,
            encoded_hidden_dim,
            decoded,
        ) = self.encoder(time_series, time, demo)
        return (
            self.assignment(encoded),
            reconstructed,
            reconstructed_time,
            encoded,
            encoded_hidden_dim,
            decoded,
        )
