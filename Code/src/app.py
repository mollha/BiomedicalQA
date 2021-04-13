from flask import Flask, render_template, request, json
from flask_bootstrap import Bootstrap
from gui.app_helpers import process_question
from os import path, walk

print('Initialising Project...')
app = Flask(__name__, template_folder='gui/templates', static_folder='gui/static')
print('Complete!\n')

# -------------------------------- CONFIGURE TEMPLATE ROUTES -------------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/handlepost', methods=['POST'])
def handle_post():
    question = request.form['question']
    context = request.form['context']  # get book id as string
    question_type = request.form['question_type']

    print("\nReceived question '{}' and context '{}'. Question type is '{}'"
          .format(question, context, question_type))

    answer = process_question(question, context, question_type)
    print("Predicted Answer:", answer)
    return json.jsonify(answer)


extra_dirs = ['/templates', "/static"]
extra_files = extra_dirs[:]
for extra_dir in extra_dirs:
    for dirname, dirs, files in walk(extra_dir):
        for filename in files:
            filename = path.join(dirname, filename)
            if path.isfile(filename):
                extra_files.append(filename)

bootstrap = Bootstrap(app)
app.run(debug=True, extra_files=extra_files)