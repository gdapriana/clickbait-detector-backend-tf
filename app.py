import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from resources.dataset import get_dataset
from resources.preprocessing import preprocessing_input


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
@cross_origin()
def hello_world():
  return jsonify({"message": "news classification"})

@app.route('/dataset')
@cross_origin()
def end_dataset():
  row = request.args.get('row')

  try:
    if row is None:
      row = 10
    row = int(row)
  except ValueError:
    return jsonify({"message": "wrong input"})

  (df,
   dataset_title,
   dataset_subtitle,
   dataset_link,
   total_dataset_count,
   taken_dataset_count,
   true_label,
   false_label) = get_dataset(num_row=row)
  return jsonify({"dataset": df,
                  "dataset_title": dataset_title,
                  "dataset_subtitle": dataset_subtitle,
                  "dataset_link": dataset_link,
                  "total_dataset": total_dataset_count,
                  "taken_dataset": taken_dataset_count,
                  "true_label": true_label,
                  "false_label": false_label
                  })

@app.route('/predict', methods=['POST'])
@cross_origin()
def end_predict():
  data = request.get_json()
  new_text = data['text']
  model = tf.keras.models.load_model('resources/model.h5')
  processed_text = preprocessing_input(new_text)
  predicted = model.predict(processed_text)
  predicted = float(predicted[0][0])
  return jsonify({"prediction": predicted})

if __name__ == '__main__':
  app.run(debug=True)