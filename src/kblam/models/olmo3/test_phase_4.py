import inspect
from transformers.models.olmo3.modeling_olmo3 import Olmo3Attention

# print(inspect.getsource(Olmo3Attention.forward))

from transformers.models.olmo3.modeling_olmo3 import eager_attention_forward

print(inspect.getsource(eager_attention_forward))