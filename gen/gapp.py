import json
from gender_bias import gender_bias
from flask import Flask, redirect, url_for, request
app = Flask(__name__)


@app.route('/output/<name>')
def output(name):
 return 'output: %s' % name


@app.route('/model', methods=['POST', 'GET'])
def login():
 if request.method == 'POST':
  user = request.form['nm']
  res = json.loads(user)
  output = gender_bias(res)
  return redirect(url_for('output', name=output))
 else:
  user = request.args.get('nm')
  res = json.loads(user)
  output = gender_bias(res)
  return redirect(url_for('output', name=output))


if __name__ == '__main__':
 app.run(debug=True)