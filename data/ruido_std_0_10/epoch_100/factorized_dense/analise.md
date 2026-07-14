# Análise: Factorized Dense Synthesizer c/ Ruído (100 épocas)

## Métricas Finais
- Test Loss: 4,18
- Test PPL: 65,07
- SacreBLEU: 7,32
- Total de épocas: 20
- Melhor época (val SacreBLEU): 19 (3,70)

## Spikes de Gradiente
- Total: 5 spikes
- Padrão: gradientes confinados (baixa magnitude)

## Análise
O Factorized Dense Synthesizer com ruído obteve 7,32 SacreBLEU, mais que o dobro do baseline sem ruído de 30épocas (4,25). O ganho relativo foi de +72,2%, o maior impacto proporcional do ruído entre todos os experimentos.

Apesar da melhora significativa, o modelo ainda apresenta qualidade de tradução muito inferior ao Dot-Product padrão, indicando que a limitação expressiva da função de score fatorada não é completamente compensada pelo ruído no gradiente.
