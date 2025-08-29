import shlex
import subprocess


subprocess.run(shlex.split("pip install pip==24.0"), check=True)
subprocess.run(
    shlex.split(
        "pip install package/onnxruntime_gpu-1.17.0-cp310-cp310-manylinux_2_28_x86_64.whl --force-reinstall --no-deps"
    ), check=True
)
subprocess.run(
    shlex.split(
        "pip install package/nvdiffrast-0.3.1.torch-cp310-cp310-linux_x86_64.whl --force-reinstall --no-deps"
    ), check=True
)


if __name__ == "__main__":
    from huggingface_hub import snapshot_download

    snapshot_download("cavargas10/Unique3D", repo_type="model", local_dir="./ckpt")

    import os
    import sys
    sys.path.append(os.curdir)
    import torch
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

import fire
import gradio as gr
from gradio_app.gradio_3dgen import create_ui as create_3d_ui
from gradio_app.all_models import model_zoo

_TITLE = """UTPL - Conversión de Imagen a objetos 3D usando IA"""
_DESCRIPTION = """
### Tesis: *"Objetos tridimensionales creados por IA: Innovación en entornos virtuales"*  
**Autor:** Carlos Vargas  
**Base técnica:** Adaptación de [Unique3D](https://wukailu.github.io/Unique3D/) (herramientas de código abierto para generación 3D)  
**Propósito educativo:** Demostraciones académicas e Investigación en modelado 3D automático    
**Agraecimiento especial a @hysts por su apoyo para el funcionamiento del demo**
"""

def launch():
    model_zoo.init_models()
        
    with gr.Blocks(
        title=_TITLE,
    ) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)
        create_3d_ui("wkl") 
    demo.queue().launch(share=True)
    
if __name__ == '__main__':
    fire.Fire(launch)