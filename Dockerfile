FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 50505

CMD ["gunicorn", "--bind", "0.0.0.0:50505", "--access-logfile", "-", "--error-logfile", "-", "src.app:create_app()"]