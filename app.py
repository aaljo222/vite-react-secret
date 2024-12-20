from flask import Flask, request, jsonify
import openai
import os

app = Flask(__name__)

# 환경 변수에서 API 키 로드
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/fine-tune", methods=["POST"])
def fine_tune():
    try:
        data = request.json
        training_file = data.get("training_file")
        model = data.get("model", "gpt-4.0-turbo")

        # Fine-Tuning API 호출
        fine_tune_job = openai.FineTuningJob.create(
            training_file=training_file,
            model=model
        )
        return jsonify({"job_id": fine_tune_job["id"]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    try:
        # Fine-Tuning 상태 확인
        job_status = openai.FineTuningJob.retrieve(id=job_id)
        return jsonify(job_status), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
