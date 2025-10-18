import os
import logging
import datetime
import jwt
from functools import wraps

from flask import Flask, request, jsonify
import joblib
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker


JWT_SECRET = "MEUSEGREDOAQUI" #acesso do token
JWT_ALGORITHM = "HS256" #tem função de assinar o token
JWT_EXP_DELTA_SECONDS = 3600 #tempo de expiração do token em segundos

logging.basicConfig(level=logging.INFO) # é usado para configurar o logging
logger = logging.getLogger("api_modelo") #cria um logger específico para este módulo

DB_URL = "sqlite:///predictions.db" #Armazena o banco de dados em predictions.db
engine = create_engine(DB_URL, echo=False) #cria a engine do SQLAlchemy
Base = declarative_base() #base para os modelos ORM que basicamente mapeiam tabelas do banco
SessionLocal = sessionmaker(bind=engine) #cria a sessão para interagir com o banco


class Prediction(Base): #tabela para armazenar predições
    __tablename__ = "predictions" #nome da tabela no banco
    id = Column(Integer, primary_key=True, autoincrement=True)
    sepal_length = Column(Float, nullable=False)
    sepal_width = Column(Float, nullable=False)
    petal_length = Column(Float, nullable=False)
    petal_width = Column(Float, nullable=False)
    predicted_class = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


Base.metadata.create_all(engine) ## Cria as tabelas no banco (em produção utilizar Alembic)

model = joblib.load("modelo_iris.pkl") #carrega o modelo treinado
logger.info("Modelo carregado com sucesso.") #log de informação

app = Flask(__name__)
predictions_cache = {}


TEST_USERNAME = "admin"
TEST_PASSWORD = "secret"


def create_token(username): #essa função cria o token JWT e define o  payload com o nome do usuário e o tempo de expiração
    payload = {
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXP_DELTA_SECONDS)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def token_required(f): #essa função é um decorador que verifica se o token JWT está presente e é válido antes de permitir o acesso ao endpoint protegido
    @wraps(f) #preserva as informações da função original
    def decorated(*args, **kwargs): #decora a função
        # pegar token do header Authorization: Bearer <token>
        # decodificar e checar expiração
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True) #força o parse do JSON mesmo que o header Content-Type não seja application/json
    username = data.get("username") #armazena o username e password enviados no corpo da requisição
    password = data.get("password") #armazena o username e password enviados no corpo da requisição
    if username == TEST_USERNAME and password == TEST_PASSWORD:
        token = create_token(username)
        return jsonify({"token": token})
    else:
        return jsonify({"error": "Credenciais inválidas"}), 401



@app.route("/predict", methods=["POST"])
@token_required
def predict():
    """
    Endpoint protegido por token para obter predição.
    Corpo (JSON):
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }
    """
    data = request.get_json(force=True)
    try:
        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])
    except (ValueError, KeyError) as e:
        logger.error("Dados de entrada inválidos: %s", e)
        return jsonify({"error": "Dados inválidos, verifique parâmetros"}), 400

    # Verificar se já está no cache
    features = (sepal_length, sepal_width, petal_length, petal_width)
    if features in predictions_cache:
        logger.info("Cache hit para %s", features)
        predicted_class = predictions_cache[features]
    else:
        # Rodar o modelo
        input_data = np.array([features])
        prediction = model.predict(input_data)
        predicted_class = int(prediction[0])
        # Armazenar no cache
        predictions_cache[features] = predicted_class
        logger.info("Cache updated para %s", features)

    # Cria uma nova sessão para acessar o banco de dados
    db = SessionLocal()
    # Cria um objeto Prediction com os dados da flor e o resultado previsto
    new_pred = Prediction(
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_length=petal_length,
        petal_width=petal_width,
        predicted_class=predicted_class
    )
    # Adiciona o objeto à sessão
    db.add(new_pred)
    # Salva (confirma) no banco de dados
    db.commit()
    # Fecha a sessão
    db.close()

    return jsonify({"prediction": predicted_class})


@app.route("/predictions", methods=["GET"])
@token_required
def list_predictions():
    """
    Lista as predições armazenadas no banco.
    Parâmetros opcionais (via query string):
      - limit (int): quantos registros retornar, padrão 10
      - offset (int): a partir de qual registro começar, padrão 0
    Exemplo:
      /predictions?limit=5&offset=10
    """
    limit = int(request.args.get("limit", 10))
    offset = int(request.args.get("offset", 0))
    db = SessionLocal()
    preds = db.query(Prediction).order_by(Prediction.id.desc()).limit(limit).offset(offset).all()
    db.close()
    results = []
    for p in preds:
        results.append({
            "id": p.id,
            "sepal_length": p.sepal_length,
            "sepal_width": p.sepal_width,
            "petal_length": p.petal_length,
            "petal_width": p.petal_width,
            "predicted_class": p.predicted_class,
            "created_at": p.created_at.isoformat()
        })
    return jsonify(results)


@app.route("/")
def route_hello():
    return jsonify({"message": "Hello, world!"})

if __name__ == "__main__":
    app.run(debug=True)