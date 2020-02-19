FROM python:3
WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV LAADS_API_KEY 889C7E56-FA7A-11E9-8719-DD4D775EF21F

COPY . .
CMD ["python","mai      n_rk.py"]