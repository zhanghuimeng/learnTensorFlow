from flask import Flask
from flask import render_template
from flask import redirect
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired

import subprocess
import shutil, os

from qe.qe_test import test as birnn_test


app = Flask(__name__)
app.config['SECRET_KEY'] = 'any secret string'
default_src = "If you are creating multiple files , you can enter common metadata for all of the files ."
default_mt = "Wenn Sie mehrere Dateien erstellen , können Sie die allgemeinen Metadaten für alle Dateien eingeben ."

birnn_model_dir = "model/qe/qe.ckpt-3220"
kiwi_model_dir = "model/en_de.smt_models/estimator/target_1"
kiwi_out_dir = "tmp_out"
kiwi_src_file = "temp_src.txt"
kiwi_mt_file = "temp_mt.txt"
kiwi_command = "kiwi predict --config demo/predict.yaml " \
               "--load-model %s/model.torch " \
               "--output-dir %s " \
               "--gpu-id -1 " \
               "--test-source %s " \
               "--test-target %s "

class QeForm(FlaskForm):
    src = StringField('src', validators=[DataRequired()], default=default_src)
    mt = StringField('mt', validators=[DataRequired()], default=default_mt)

@app.route('/', methods=('GET', 'POST'))
def submit():
    form = QeForm()
    if form.validate_on_submit():
        print(form.src.data)
        print(form.mt.data)
        # birnn
        vocab = ["data/qe-2017/src.vocab", "data/qe-2017/tgt.vocab"]
        test = [[form.src.data], [form.mt.data], ["0.0"]]
        model = birnn_model_dir
        pred = birnn_test(vocab=vocab, test=test, model_addr=model)
        # kiwi
        # write to files
        with open(kiwi_src_file, 'w') as f:
            f.write(form.src.data)
        with open(kiwi_mt_file, 'w') as f:
            f.write(form.mt.data)
        command = kiwi_command % (kiwi_model_dir, kiwi_out_dir, kiwi_src_file, kiwi_mt_file)
        print(command)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()
        print(process.returncode)
        kiwi_score = None
        if process.returncode == 0:
            with open(kiwi_out_dir + "/sentence_scores", 'r') as f:
                kiwi_score = float(f.read())
        os.remove(kiwi_src_file)
        os.remove(kiwi_mt_file)
        shutil.rmtree(kiwi_out_dir)
        return render_template('index.html', form=form, birnn_qe_score=pred[0][0],
                               openkiwi_qe_score=kiwi_score)
    return render_template('index.html', form=form)
