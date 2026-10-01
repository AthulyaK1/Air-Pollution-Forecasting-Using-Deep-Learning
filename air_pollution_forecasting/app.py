from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from utils.predictor import load_model_and_scaler, predict_future
from utils.data_loader import get_sample_data, get_historical_stats
import json
import os

app = Flask(__name__)
app.secret_key = 'airsense_secret_key_2026'

USERS_FILE = 'data/users.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {
        'admin':   {'password': 'admin123', 'role': 'Admin'},
        'athulya': {'password': 'mca2026',  'role': 'Student'},
        'guest':   {'password': 'guest123', 'role': 'Guest'},
    }

def save_users(users):
    os.makedirs('data', exist_ok=True)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

model, scaler = None, None

def get_model():
    global model, scaler
    if model is None:
        model, scaler = load_model_and_scaler()
    return model, scaler

# ── Auth Routes ───────────────────────────────────────────────────────────────
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        users = load_users()
        if username in users and users[username]['password'] == password:
            session['username'] = username
            session['role'] = users[username].get('role', 'User')
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid username or password. Please try again.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ── Account Routes ────────────────────────────────────────────────────────────
@app.route('/create_account', methods=['POST'])
def create_account():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    role = data.get('role', 'Student')
    if not username or not password:
        return jsonify({'success': False, 'message': 'All fields required.'})
    users = load_users()
    if username in users:
        return jsonify({'success': False, 'message': 'Username already exists.'})
    if len(password) < 6:
        return jsonify({'success': False, 'message': 'Password must be at least 6 characters.'})
    users[username] = {'password': password, 'role': role}
    save_users(users)
    return jsonify({'success': True, 'message': 'Account created for "' + username + '". You can now login.'})

@app.route('/reset_password', methods=['POST'])
def reset_password():
    data = request.get_json()
    user = data.get('username', '').strip()
    cur  = data.get('current_password', '').strip()
    new  = data.get('new_password', '').strip()
    users = load_users()
    if user not in users:
        return jsonify({'success': False, 'message': 'Username not found.'})
    if users[user]['password'] != cur:
        return jsonify({'success': False, 'message': 'Current password incorrect.'})
    if len(new) < 6:
        return jsonify({'success': False, 'message': 'New password must be 6+ characters.'})
    users[user]['password'] = new
    save_users(users)
    return jsonify({'success': True, 'message': 'Password updated. Please login.'})

@app.route('/reset_username', methods=['POST'])
def reset_username():
    data = request.get_json()
    cur  = data.get('current_username', '').strip()
    pwd  = data.get('password', '').strip()
    new  = data.get('new_username', '').strip()
    users = load_users()
    if cur not in users:
        return jsonify({'success': False, 'message': 'Username not found.'})
    if users[cur]['password'] != pwd:
        return jsonify({'success': False, 'message': 'Incorrect password.'})
    if new in users:
        return jsonify({'success': False, 'message': 'Username already taken.'})
    users[new] = users[cur]
    del users[cur]
    save_users(users)
    return jsonify({'success': True, 'message': 'Username updated to "' + new + '".'})

# ── Page Routes ───────────────────────────────────────────────────────────────
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['username'])

@app.route('/upload')
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('upload.html', username=session['username'])

@app.route('/predict')
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('predict.html', username=session['username'])

@app.route('/report')
def report():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('report.html', username=session['username'])

@app.route('/evaluation')
def evaluation():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('evaluation.html', username=session['username'])

# ── API Routes ────────────────────────────────────────────────────────────────
@app.route('/api/forecast', methods=['POST'])
def forecast():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.json
    pollutant = data.get('pollutant', 'PM2.5')
    steps = int(data.get('steps', 24))
    m, s = get_model()
    predictions = predict_future(m, s, pollutant, steps)
    return jsonify({'predictions': predictions, 'pollutant': pollutant, 'steps': steps})

@app.route('/api/historical')
def historical():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    pollutant = request.args.get('pollutant', 'PM2.5')
    stats = get_historical_stats(pollutant)
    return jsonify(stats)

@app.route('/api/current')
def current():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    data = get_sample_data()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
