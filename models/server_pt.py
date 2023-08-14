import copy
import models.server
from models.baseline_constants import TENSORFLOW_OR_PYTORCH
from collections import OrderedDict
import torch


class Server(models.server.Server):
    def __init__(self, client_model):
        assert TENSORFLOW_OR_PYTORCH == 'PT', f'This class implements pytorch server. Configuration set to' \
                                              f' TENSORFLOW_OR_PYTORCH={TENSORFLOW_OR_PYTORCH}'

        self.client_model = client_model
        self.model = copy.copy(client_model)
        self.selected_clients = []
        self.updates = []

    def update_model(self):
        params = OrderedDict()
        for n, p in self.model.named_parameters():
            params[n] = torch.zeros_like(p.data)

        for c in self.selected_clients:
            for n, p in c.model.named_parameters():
                params[n] += p.data

        # average parameters
        for n, p in params.items():
            params[n] = p / len(self.selected_clients)
        # update new parameters
        self.model.load_state_dict(params, strict=False)
        self.updates = []

    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model.set_params(self.model)
        return self.client_model.save(path)
