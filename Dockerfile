FROM python:3.10-slim-buster

WORKDIR /app

# Install TA-Lib dependencies
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     wget     && rm -rf /var/lib/apt/lists/*

RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz &&     tar -xvzf ta-lib-0.4.0-src.tar.gz &&     cd ta-lib/ &&     ./configure --prefix=/usr &&     make &&     make install &&     cd .. &&     rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/

CMD ["python", "-m", "src.main", "--mode", "trade"]
