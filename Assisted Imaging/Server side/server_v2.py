import os
import argparse
import numpy as np

from flask import Flask
from flask import render_template, request

from options.test_options import TestOptions
from models import create_model
import util
import torch

import json

app = Flask(__name__)

@app.route("/", methods=['POST'])
def generate_synthetic():
    global model, useCuda
    data = json.loads(request.data)#.decode('utf-8'))
    model.set_input(data) # unpack data from data loader
    model.compute_std_map()
    model.next_acquisition()
    model.forward()
    dict_data = {'nextSTED':model.nextSTED, 'std_map':model.std_map.tolist(), 'synthetic_STED':model.fake_B.tolist(), 'seg_fibers':model.seg_fibers_array.tolist(), 'seg_real_STED':model.seg_real_STED.tolist()}
    return json.dumps(dict_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Server for SR-Generation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--port', help='port if using virtual net', type=int, default=5000)
    parser.add_argument('--cuda', help='use GPU or not', action="store_true", default=True)
    parser.add_argument('-v', '--verbose', help='print information', action="store_true", default=False)
    args = parser.parse_args()
    useCuda = args.cuda

    opt = TestOptions().parse()  # get test options
    print("Defining model...")
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    print("Model was successfully loaded!")

    app.run(host='0.0.0.0', port=args.port, debug=False, use_reloader=False) 
