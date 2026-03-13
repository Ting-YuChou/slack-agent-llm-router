FROM flink:1.20-scala_2.12-java17

ARG FLINK_KAFKA_CONNECTOR_VERSION=3.3.0-1.20
ARG KAFKA_CLIENTS_VERSION=3.4.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        openjdk-17-jdk-headless \
        python3 \
        python3-dev \
        python3-pip \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && curl -fsSL \
        "https://repo1.maven.org/maven2/org/apache/flink/flink-connector-kafka/${FLINK_KAFKA_CONNECTOR_VERSION}/flink-connector-kafka-${FLINK_KAFKA_CONNECTOR_VERSION}.jar" \
        -o /opt/flink/lib/flink-connector-kafka.jar \
    && curl -fsSL \
        "https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/${KAFKA_CLIENTS_VERSION}/kafka-clients-${KAFKA_CLIENTS_VERSION}.jar" \
        -o /opt/flink/lib/kafka-clients.jar \
    && JAVA_INCLUDE_DIR="$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")/include" \
    && mkdir -p /opt/java/openjdk \
    && rm -rf /opt/java/openjdk/include \
    && ln -sfn "${JAVA_INCLUDE_DIR}" /opt/java/openjdk/include \
    && ln -sfn "${JAVA_INCLUDE_DIR}/linux" /opt/java/openjdk/include/linux \
    && pip3 install --no-cache-dir apache-flink==1.20.0 \
    && rm -rf /var/lib/apt/lists/*
