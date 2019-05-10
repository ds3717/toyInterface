#!/usr/bin/env python2.7
import os
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, g, redirect, Response, flash, session, url_for, send_from_directory
from clean_data import *
from predict_label import *


tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/uploads/'
DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/downloads/'
ALLOWED_EXTENSIONS = {'xls', 'xlsx'}
app = Flask(__name__, template_folder=tmpl_dir)
app.secret_key = 'some_secret'
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER


def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    if 'file' not in request.files:
      print('No file attached in request')
      return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
      print('No file selected')
      return redirect(request.url)
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      process_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename)
      return redirect(url_for('uploaded_file', filename=filename))
  return render_template('index.html')


def process_file(path, filename):
  d_data = delimited_data(path)
  predict_data = predict_labels(d_data)
  writer = pd.ExcelWriter(app.config['DOWNLOAD_FOLDER'] + 'pred_' + filename, engine='xlsxwriter')
  predict_data.to_excel(writer, index=False, encoding='utf-8', sheet_name='Sheet1')
  writer.save()


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], 'pred_' + filename, as_attachment=True)


if __name__ == "__main__":
  import click

  @click.command()
  @click.option('--debug', is_flag=True)
  @click.option('--threaded', is_flag=True)
  @click.argument('HOST', default='0.0.0.0')
  @click.argument('PORT', default=8111, type=int)
  def run(debug, threaded, host, port):
    """
    This function handles command line parameters.
    Run the server using

        python server.py

    Show the help text using

        python server.py --help

    """

    HOST, PORT = host, port
    print ("running on %s:%d" % (HOST, PORT))
    app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)


  run()
