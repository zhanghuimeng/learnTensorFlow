from flask import Flask
from flask import render_template
from flask import redirect
from flask_wtf import FlaskForm
from wtforms import StringField
from  wtforms.validators import DataRequired


app = Flask(__name__)
app.config['SECRET_KEY'] = 'any secret string'

class QeForm(FlaskForm):
    src = StringField('src', validators=[DataRequired()])
    mt = StringField('mt', validators=[DataRequired()])

@app.route('/', methods=('GET', 'POST'))
def submit():
    form = QeForm()
    if form.validate_on_submit():
        print(form.src.data)
        print(form.src.data)
        return render_template('index.html', form=form)
    return render_template('index.html', form=form)
