from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app)

MLFLOW_SERVER = 'http://localhost:5000'

@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy(path):
    url = f'{MLFLOW_SERVER}/api/{path}'

    if request.method == 'OPTIONS':
        return '', 200

    forward_headers = {}
    for k, v in request.headers.items():
      lk = k.lower()
      if lk in ('authorization', 'content-type'):
        forward_headers[k] = v

    try:
        if request.method == 'POST':
            resp = requests.post(url, json=request.get_json(silent=True), headers=forward_headers)
        elif request.method == 'PUT':
            resp = requests.put(url, json=request.get_json(silent=True), headers=forward_headers)
        elif request.method == 'DELETE':
            resp = requests.delete(url, headers=forward_headers)
        else:  # GET
            resp = requests.get(url, params=request.args, headers=forward_headers)
        
        return resp.content, resp.status_code, dict(resp.headers)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/artifact-content', methods=['GET', 'OPTIONS'])
def artifact_content():
    """
    Return content of an artifact for a run using local file store.
    Query params:
      - run_id or run_uuid
      - path (relative path under artifacts/)
    """
    if request.method == 'OPTIONS':
        return '', 200

    run_id = request.args.get('run_id') or request.args.get('run_uuid')
    rel_path = request.args.get('path')

    if not run_id or not rel_path:
        return jsonify({'error': 'run_id/run_uuid and path are required'}), 400

    try:
        resp = requests.get(f'{MLFLOW_SERVER}/api/2.0/mlflow/runs/get', params={'run_id': run_id})
        if not resp.ok:
            return jsonify({'error': f'Failed to fetch run info: {resp.status_code}'}), resp.status_code
        run = resp.json().get('run')

        if not run:
            return jsonify({'error': 'Run not found'}), 404

        artifact_uri = run['info'].get('artifact_uri', '')
        print(artifact_uri)
        parsed = urlparse(artifact_uri)
        print(parsed)

        if parsed.scheme != 'file':
            return jsonify({'error': f'Unsupported artifact scheme: {parsed.scheme}'}), 400
        print('hello 1')

        base_dir = parsed.path
        file_path = os.path.abspath(os.path.join(base_dir, rel_path))
        print('hello 2')

        if not os.path.isfile(file_path):
            return jsonify({'error': f'Artifact not found: {file_path}'}), 404
        print('hello 3')

        ext = os.path.splitext(file_path)[1].lower()
        print('hello 4')
        content_type = 'application/octet-stream'
        print('hello 5')
        if ext in ('.csv', '.txt'):
            content_type = 'text/csv'
        elif ext in ('.json',):
            content_type = 'application/json'

        with open(file_path, 'rb') as f:
            data = f.read()
        print(data)
        return data, 200, {'Content-Type': content_type}
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 CORS Proxy running on http://localhost:8080")
    print("📡 Forwarding requests to MLflow at http://localhost:5000")
    app.run(host='0.0.0.0', port=8080, debug=True)