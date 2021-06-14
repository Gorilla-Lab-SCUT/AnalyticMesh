""" python server
    run example:
        python server.py
"""

import os
import uuid
import math
import json
import base64
import asyncio
import argparse
import websockets
from onnx_vis import OnnxToPng
from AnalyticMesh import load_model, AnalyticMarching, simplify, estimate_am_time
from AnalyticMesh.backend.main import SPHERE_TRACING_DICT, GRADIENT_DESCENT_DICT, DICHOTOMY_DICT
import threading
from kill_thread import stop_thread

DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "temp")
os.makedirs(DIR, exist_ok=True)

######################################################################################################
def get_rand_filename():
    """ get random filename
    """
    filename = str(uuid.uuid4())  # e.g. 78302d95-7bbc-48d7-bf07-63194dadaaba
    return filename


def update_dict(new_dict):
    """ update configs dict
    """
    if new_dict["method"] == "dichotomy":
        default_dict = DICHOTOMY_DICT.copy()
    elif new_dict["method"] == "sphere_tracing":
        default_dict = SPHERE_TRACING_DICT.copy()
    elif new_dict["method"] == "gradient_descent":
        default_dict = GRADIENT_DESCENT_DICT.copy()
    else:
        raise Exception("unknown triggering method: " + new_dict["method"])
    default_dict["args"].update(new_dict["args"])
    updated_dict = dict(method=default_dict["method"],
                        args=default_dict["args"])
    return updated_dict


def parseData(_dict):
    """ transform type
    """
    for key in _dict:
        if key == 'init_num' or key == 'iter_max':
            _dict[key] = int(_dict[key])
        else:
            _dict[key] = float(_dict[key])
    return _dict


async def main(ws, path):
    """ main function
    """
    private_memory = {}
    while ws.closed is False:
        try:
            data = await ws.recv()
            try:
                json_data = json.loads(data)
                if 'action' not in json_data or 'data' not in json_data:
                    raise Exception('error data')
            except:
                print(ws, 'error data:' + str(data))
                continue
            if json_data['action'] == 'import':
                await import_event(ws, json_data['data'].replace(r'data:application/octet-stream;base64,', ''),
                                   private_memory)
            elif json_data['action'] == 'submit':
                await submit_event(ws, json_data['data'], private_memory)
            elif json_data['action'] == 'simplify':
                await simplify_event(ws, json_data['data'], private_memory)
        except Exception as e:
            print(ws, 'error', e)
            await ws.close()
    # clean private memory
    if 'thread' in private_memory:
        for t in private_memory['thread']:
            if hasattr(t, 'is_alive') and t.is_alive():
                stop_thread(t)
    if 'ply_path' in private_memory and os.path.exists(private_memory['ply_path']):
        os.unlink(private_memory['ply_path'])
    if 'simplified_ply_path' in private_memory and os.path.exists(private_memory['simplified_ply_path']):
        os.unlink(private_memory['simplified_ply_path'])
    del private_memory
    return


async def import_event(ws, data, private_memory):
    OTP = OnnxToPng()
    model = base64.b64decode(data)
    private_memory['model'] = model
    png_data = OTP.run(model)
    return_data = json.dumps({
        'callback': 'import',
        'data': base64.b64encode(png_data).decode('utf-8')
    })
    return await ws.send(return_data)


async def submit_event(ws, data, private_memory):
    # check state
    if 'model' not in private_memory or private_memory['model'] == b'':
        return_data = json.dumps({
            'callback': 'submit',
            'data': {
                'success': 0,
                'msg': 'please upload model firstly'
            }
        })
        return await ws.send(return_data)
    if 'iso' not in data or 'triggering' not in data:
        return_data = json.dumps({
            'callback': 'submit',
            'data': {
                'success': 0,
                'msg': 'submit data is error'
            }
        })
        return await ws.send(return_data)
    ply_path = os.path.join(DIR, get_rand_filename() + ".ply")
    if 'ply_path' in private_memory and os.path.exists(private_memory['ply_path']):
        os.unlink(private_memory['ply_path'])
    private_memory['ply_path'] = ply_path
    parseData(data['triggering']['args'])
    # do analytic marching
    model = load_model(private_memory['model'])

    return_data = json.dumps({
        'callback': 'submit',
        'data': {
            'success': 1,
            'type': 'estimate time',
            'long': estimate_am_time(model)
        }
    })
    await ws.send(return_data)

    def run_model(model, ply_path, data):
        asyncio.set_event_loop(asyncio.new_event_loop())

        result = AnalyticMarching(model, ply_path,
                                  iso=float(data["iso"]),
                                  init_configs=update_dict(data["triggering"]),
                                  save_polymesh=False)
        return_data = json.dumps({
            'callback': 'submit',
            'data': {
                'success': 1,
                'type': 'result',
                'data': result
            }
        })
        asyncio.get_event_loop().run_until_complete(ws.send(return_data))
        return

    t = threading.Thread(target=run_model, args=(model, ply_path, data,), daemon=True)
    t.start()
    if 'thread' not in private_memory:
        private_memory['thread'] = [t]
    else:
        private_memory['thread'].append(t)

    while t.is_alive():
        return_data = json.dumps({
            'callback': 'submit',
            'data': {
                'success': 1,
                'type': 'running'
            }
        })
        await ws.send(return_data)
        t.join(timeout=1)

    return_data = json.dumps({
        'callback': 'submit',
        'data': {
            'success': 1,
            'type': 'download model size',
            'long': math.ceil(os.path.getsize(ply_path) / 100 / 1024),
            'real_size': os.path.getsize(ply_path)
        }
    })
    await ws.send(return_data)

    # download
    f = open(ply_path, 'rb')
    while True:
        data = f.read(100 * 1024)
        if not data:
            break
        return_data = json.dumps({
            'callback': 'submit',
            'data': {
                'success': 1,
                'type': 'downloading',
                'data': base64.b64encode(data).decode('utf-8')
            }
        })
        await ws.send(return_data)
    return_data = json.dumps({
        'callback': 'submit',
        'data': {
            'success': 1,
            'type': 'finish'
        }
    })
    return await ws.send(return_data)


async def simplify_event(ws, data, private_memory):
    if 'ply_path' not in private_memory or private_memory['ply_path'] == "" or os.path.exists(
            private_memory['ply_path']) is False:
        return_data = json.dumps({
            'callback': 'simplify',
            'data': {
                'success': 0,
                'msg': 'please submit firstly'
            }
        })
        return await ws.send(return_data)
    try:
        perc = float(data)
        if perc < 0:
            perc = 0
        elif perc > 1:
            perc = 1
    except e:
        perc = 1

    mesh_simplified_path = os.path.join(DIR, get_rand_filename() + ".ply")
    # remove old simplified
    if 'simplified_ply_path' in private_memory and os.path.exists(private_memory['simplified_ply_path']):
        os.unlink(private_memory['simplified_ply_path'])
    private_memory['simplified_ply_path'] = mesh_simplified_path
    print('ply:', private_memory['ply_path'])
    print('simplified:', mesh_simplified_path)

    def run_model(ply_path, simplified_path, perc):
        simplify(ply_path, simplified_path, target_perc=perc)
        return

    t = threading.Thread(target=run_model, args=(private_memory['ply_path'], mesh_simplified_path, perc,), daemon=True)
    t.start()
    private_memory['thread'].append(t)

    while t.is_alive():
        return_data = json.dumps({
            'callback': 'simplify',
            'data': {
                'success': 1,
                'type': 'running'
            }
        })
        await ws.send(return_data)
        t.join(timeout=1)

    return_data = json.dumps({
        'callback': 'simplify',
        'data': {
            'success': 1,
            'type': 'download model size',
            'long': math.ceil(os.path.getsize(mesh_simplified_path) / 100 / 1024),
            'real_size': os.path.getsize(mesh_simplified_path)
        }
    })
    await ws.send(return_data)

    # download
    f = open(mesh_simplified_path, 'rb')
    while True:
        data = f.read(100 * 1024)
        if not data:
            break
        return_data = json.dumps({
            'callback': 'simplify',
            'data': {
                'success': 1,
                'type': 'downloading',
                'data': base64.b64encode(data).decode('utf-8')
            }
        })
        await ws.send(return_data)
    return_data = json.dumps({
        'callback': 'simplify',
        'data': {
            'success': 1,
            'type': 'finish'
        }
    })
    return await ws.send(return_data)


#######################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()

    PORT = args.port

    server = websockets.serve(main, "0.0.0.0", PORT, max_size=None, ping_interval=None)
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()
