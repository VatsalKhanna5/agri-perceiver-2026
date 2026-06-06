fp = '/Data1/ece_23104085/hf_cache/modules/transformers_modules/OpenGVLab/InternVL2-8B/6fb9ad6924f69424e57fab2ab061d707688f0296/modeling_internlm2.py'
c = open(fp).read()
changed = []

# 1. Add GenerationMixin import
old1 = 'from transformers.modeling_utils import PreTrainedModel'
new1 = old1 + '\nfrom transformers.generation.utils import GenerationMixin'
if 'from transformers.generation.utils import GenerationMixin' not in c:
    c = c.replace(old1, new1, 1)
    changed.append('import')

# 2. Fix class definition
old2 = 'class InternLM2ForCausalLM(InternLM2PreTrainedModel):'
new2 = 'class InternLM2ForCausalLM(InternLM2PreTrainedModel, GenerationMixin):'
if old2 in c:
    c = c.replace(old2, new2, 1)
    changed.append('class_def')

# 3. Fix InternLM2Model.forward - past_key_values_length (line ~890)
old3 = '            past_key_values_length = past_key_values[0][0].shape[2]'
new3 = (
    '            if hasattr(past_key_values, "get_seq_length"):\n'
    '                past_key_values_length = past_key_values.get_seq_length()\n'
    '            else:\n'
    '                past_key_values_length = past_key_values[0][0].shape[2]'
)
if old3 in c:
    c = c.replace(old3, new3, 1)
    changed.append('model_fwd')

# 4. Fix prepare_inputs_for_generation - handle empty DynamicCache
old4 = '        if past_key_values is not None:\n            past_length = past_key_values[0][0].shape[2]'
new4 = (
    '        if past_key_values is not None:\n'
    '            if hasattr(past_key_values, "get_seq_length"):\n'
    '                past_length = past_key_values.get_seq_length()\n'
    '                if past_length == 0:\n'
    '                    past_key_values = None\n'
    '            else:\n'
    '                past_length = past_key_values[0][0].shape[2]'
)
if old4 in c:
    c = c.replace(old4, new4, 1)
    changed.append('prepare_inputs')

open(fp, 'w').write(c)
print('Patched:', changed)
