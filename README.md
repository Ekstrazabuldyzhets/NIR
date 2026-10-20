О проекте
Модульный пайплайн трансферного обучения для прогнозирования State of Health (SoH) аккумуляторных батарей по временным рядам. Цель - перенести знания нейросети с домена Source на Target с малым объемом данных. Код разделен на слабосвязанные модули с реестром методов, что позволяет добавлять новые архитектуры и честно сравнивать стратегии адаптации.

Данные (Target): Battery charging data of on-road electric vehicles (20 EVs). 20 коммерческих электромобилей (BAIC EU500, CATL NCM 145 Ah), ~29 месяцев полевых данных через CAN (частота 8 с). Цитирование: Deng et al., Applied Energy, 2023, DOI: 10.1016/j.apenergy.2023.120954. https://github.com/BatICM/battery-charging-data-of-on-road-electric-vehicles 

Архитектура и методы
В основе - Encoder-Decoder BiLSTM + Attention с коррекцией остатков через Multi-Scale GPR. Метод адаптации переключается одной строкой в конфиге: none (zero-shot), freeze_enc (поэтапная заморозка), bridge (латентный мост на Random Forest) и dann (Domain-Adversarial Neural Network с GRL). Пайплайн включает MMD-анализ доменов, раздельную нормализацию, CORAL-выравнивание и кросс-валидацию по автомобилям.
