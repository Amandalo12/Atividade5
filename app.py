# ============================================================
# App Streamlit: Projeção da conta de energia do ar-condicionado
# Como rodar:
#   pip install streamlit tensorflow pandas numpy
#   streamlit run app.py
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ------------------------------------------------------------
# 1) Configuração da página
# ------------------------------------------------------------
st.set_page_config(page_title="Projeção do Ar-Condicionado", page_icon="❄️")
st.title("❄️ Projeção da conta do ar-condicionado")
st.write(
    "Informe seu uso e veja quanto o aparelho deve custar no mês, "
    "calculado e projetado com **TensorFlow**."
)

# ------------------------------------------------------------
# 2) Entradas do usuário
# ------------------------------------------------------------
st.sidebar.header("Dados de uso")

horas_por_dia = st.sidebar.number_input(
    "Horas de uso por dia", min_value=0.0, max_value=24.0, value=6.0, step=0.5
)
potencia_w = st.sidebar.number_input(
    "Potência elétrica do aparelho (W)",
    min_value=100, max_value=5000, value=1000, step=50,
    help="Veja na etiqueta do aparelho. Um modelo de 12.000 BTU costuma ficar perto de 1000 W.",
)
tarifa_kwh = st.sidebar.number_input(
    "Tarifa (R$ por kWh)",
    min_value=0.10, max_value=5.00, value=0.90, step=0.05,
    help="Confira o valor na sua conta de luz (já com impostos).",
)
dias_no_mes = st.sidebar.number_input(
    "Dias no mês", min_value=28, max_value=31, value=31
)
dias_passados = st.sidebar.slider(
    "Dias já decorridos no mês", min_value=2, max_value=int(dias_no_mes), value=16
)

# ------------------------------------------------------------
# 3) Cálculo do gasto diário com operações do TensorFlow
#    Tensores são "vetores" de números que o TF sabe processar.
# ------------------------------------------------------------
potencia_kw = potencia_w / 1000.0  # converte W para kW

# Um tensor com as horas de uso de cada dia que já passou
horas_tensor = tf.constant([horas_por_dia] * dias_passados, dtype=tf.float32)

# kWh consumidos por dia = horas × potência (kW)
kwh_dia = horas_tensor * potencia_kw

# Custo diário em reais = kWh × tarifa
custo_dia = kwh_dia * tarifa_kwh

# tf.cumsum soma acumulando: dia 1, dia 1+2, dia 1+2+3...
# Isso dá a "evolução do gasto" ao longo do mês
gasto_acumulado = tf.cumsum(custo_dia)

# ------------------------------------------------------------
# 4) Projeção com uma rede neural mínima (regressão linear no Keras)
#    Ela aprende a reta "gasto acumulado x dia" e a estende até o fim do mês.
# ------------------------------------------------------------
def projetar_gasto(gasto_real: np.ndarray, total_dias: int) -> np.ndarray:
    """Treina um modelo linear e devolve o gasto previsto para todos os dias."""
    tf.random.set_seed(42)  # resultados reproduzíveis

    dias_reais = np.arange(1, len(gasto_real) + 1, dtype="float32")
    todos_os_dias = np.arange(1, total_dias + 1, dtype="float32")

    # Normalizar os dados ajuda o treino a convergir rápido
    escala_x = float(total_dias)
    escala_y = max(float(gasto_real.max()), 1e-6)  # evita divisão por zero

    x = (dias_reais / escala_x).reshape(-1, 1)
    y = (gasto_real / escala_y).reshape(-1, 1)

    # Modelo: uma única camada Dense com 1 neurônio, ou seja, y = a*x + b
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(1),
    ])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss="mse")
    modelo.fit(x, y, epochs=300, verbose=0)  # treino silencioso

    # Previsão para o mês inteiro (desfaz a normalização)
    previsao = modelo.predict((todos_os_dias / escala_x).reshape(-1, 1), verbose=0)
    return np.clip(previsao.flatten() * escala_y, 0, None)  # sem valores negativos


# O botão evita retreinar o modelo a cada clique em qualquer campo
if st.button("Calcular projeção", type="primary"):
    with st.spinner("Treinando o modelo..."):
        previsao = projetar_gasto(gasto_acumulado.numpy(), int(dias_no_mes))

    # --------------------------------------------------------
    # 5) Montar os dados do gráfico
    #    NaN nos dias futuros deixa a linha "Gasto real" parar no dia atual.
    # ------------------------------------------------------
    gasto_real_completo = np.full(int(dias_no_mes), np.nan)
    gasto_real_completo[:dias_passados] = gasto_acumulado.numpy()

    df = pd.DataFrame(
        {"Gasto real (R$)": gasto_real_completo, "Projeção (R$)": previsao},
        index=pd.Index(range(1, int(dias_no_mes) + 1), name="Dia do mês"),
    )

    # --------------------------------------------------------
    # 6) Resultados na tela
    # --------------------------------------------------------
    total_projetado = float(previsao[-1])
    gasto_ate_agora = float(gasto_acumulado[-1])
    kwh_total = float(tf.reduce_sum(kwh_dia))

    col1, col2, col3 = st.columns(3)
    col1.metric("Gasto até hoje", f"R$ {gasto_ate_agora:,.2f}")
    col2.metric("Projeção do mês", f"R$ {total_projetado:,.2f}")
    col3.metric("kWh até hoje", f"{kwh_total:,.1f}")

    st.subheader("Evolução do gasto no mês")
    st.line_chart(df)  # gráfico de linha pedido

    st.caption(
        "Estimativa apenas do ar-condicionado, sem bandeiras tarifárias "
        "ou outros aparelhos. O consumo real varia com a temperatura e o modelo."
    )
else:
    st.info("Preencha os dados na barra lateral e clique em **Calcular projeção**.")