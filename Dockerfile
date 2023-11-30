FROM python:3.9

COPY . /app

WORKDIR "/app"

RUN pip install fastapi uvicorn

EXPOSE 80

CMD ["uvicorn", "new_app:app", "--reload", "--port", "80"]
