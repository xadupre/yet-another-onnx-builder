"""
.. _l-plot-input-observer-tiny-llm:

Export a LLM with InputObserver (with Tiny-LLM)
===============================================

The main issue when exporting a LLM is the example on HuggingFace is
based on method generate but we only need to export the forward method.
Example :ref:`l-plot-input-observer-transformers` gives details on how to guess
dummy inputs and dynamic shapes to do so.
Let's see how to simplify that.

Dummy Example
+++++++++++++

Let's use the example provided on
`arnir0/Tiny-LLM <https://huggingface.co/arnir0/Tiny-LLM>`_.
"""

import pandas
from transformers import AutoModelForCausalLM, AutoTokenizer
from yobx import doc
from yobx.helpers import string_type
from yobx.helpers.rt_helper import onnx_generate
from yobx.torch import (
    register_flattening_functions,
    apply_patches_for_model,
    to_onnx,
    InputObserver,
)

MODEL_NAME = "arnir0/Tiny-LLM"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def generate_text(
    prompt,
    model,
    tokenizer,
    max_length=50,
    temperature=0.01,
    top_k=50,
    top_p=0.95,
    do_sample=True,
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# Define your prompt
prompt = "Continue: it rains, what should I do?"
generated_text = generate_text(prompt, model, tokenizer)
print("-----------------")
print(generated_text)
print("-----------------")

# %%
# Replace forward method
# ++++++++++++++++++++++
#
# We first capture inputs and outputs with an :class`InputObserver
# <yobx.investigate.input_observer>`.
# We also need to registers additional patches for :epkg:`transformers`.
# Then :epkg:`pytorch` knows how to flatten/unflatten inputs.


observer = InputObserver()
with register_flattening_functions(patch_transformers=True), observer(model):
    generate_text(prompt, model, tokenizer)

print(f"number of stored inputs: {len(observer.info)}")

# %%
# Exports
# +++++++
#
# The `InputObserver` has now enough data to infer arguments and dynamic shapes.
# We need more than flattening but also patches to export the model.
# Inferred dynamic shapes looks like:
with register_flattening_functions(patch_transformers=True):
    dynamic_shapes = observer.infer_dynamic_shapes(set_batch_dimension_for=True)
    kwargs = observer.infer_arguments()

# %%
# and inferred arguments:
print("dynamic_shapes:", dynamic_shapes)
print("kwargs:", string_type(kwargs, with_shape=True))

# %%
# Let's export.

filenamec = "plot_input_observer_tiny_llm.onnx"
with (
    register_flattening_functions(patch_transformers=True),
    apply_patches_for_model(patch_torch=True, patch_transformers=True, model=model),
):
    to_onnx(
        model,
        (),
        kwargs=observer.infer_arguments(),
        dynamic_shapes=observer.infer_dynamic_shapes(set_batch_dimension_for=True),
        filename=filenamec,
    )

# %%
# Check discrepancies
# +++++++++++++++++++
#
# The model is exported into ONNX. We use again the stored inputs and outputs
# to verify the model produces the same outputs.

data = observer.check_discrepancies(filenamec, progress_bar=True)
print(pandas.DataFrame(data))


# %%
# Minimal script to export a LLM
# ++++++++++++++++++++++++++++++
#
# The following lines are a condensed copy with less comments.

# from HuggingFace
print("----------------")
MODEL_NAME = "arnir0/Tiny-LLM"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# from HuggingFace again
prompt = "Continue: it rains, what should I do?"
inputs = tokenizer(prompt, return_tensors="pt")

observer = InputObserver()

with (
    register_flattening_functions(patch_transformers=True),
    apply_patches_for_model(patch_transformers=True, model=model),
    observer(model),
):
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        do_sample=False,
        max_new_tokens=10,
    )

filename = "plot_input_observer_tiny_llm.2.onnx"
with (
    register_flattening_functions(patch_transformers=True),
    apply_patches_for_model(patch_torch=True, patch_transformers=True, model=model),
):
    to_onnx(
        model,
        (),
        kwargs=observer.infer_arguments(),
        filename=filename,
        dynamic_shapes=observer.infer_dynamic_shapes(set_batch_dimension_for=True),
    )

data = observer.check_discrepancies(filename, progress_bar=True)
print(pandas.DataFrame(data))

# %%
# %%
# ONNX Prompt
# +++++++++++
#
# :func:`onnx_generate <yobx.helpers.rt_helper.onnx_generate>` runs the
# exported ONNX model in a greedy auto-regressive loop, feeding the
# *present* key/value tensors back as *past* key/values on every subsequent
# call, just like the HuggingFace ``generate`` method.

onnx_tokens = onnx_generate(
    filenamec,
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    eos_token_id=model.config.eos_token_id,
    max_new_tokens=50,
)
onnx_generated_text = tokenizer.decode(onnx_tokens[0], skip_special_tokens=True)
print("-----------------")
print(onnx_generated_text)
print("-----------------")

# %%
doc.save_fig(doc.plot_dot(filename), f"{filename}.png", dpi=400)
