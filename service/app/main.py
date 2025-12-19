"""Приложение Fast API для рекомендательной системы банковских продуктов."""
from fastapi import FastAPI, Body
from .handler import FastApiHandler
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter, Gauge
import json
from typing import List, Dict, Any

# Создаем приложение Fast API
app = FastAPI(
    title="Bank Product Recommendation API",
    description="API для рекомендации банковских продуктов на основе Random Forest",
    version="1.0.0"
)

# Инициализируем инструментарий Prometheus
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Создаем обработчик запросов для API
handler = FastApiHandler()

# Метрики Prometheus
main_app_predictions = Histogram(
    "bank_app_recommendation_probability",
    "Гистограмма вероятностей рекомендаций",
    buckets=(0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
)

main_app_counter = Counter(
    "bank_app_predictions_total", 
    "Счетчик всех предсказаний"
)

main_app_users = Counter(
    "bank_app_unique_users",
    "Счетчик уникальных пользователей"
)

model_loaded_gauge = Gauge(
    "bank_app_model_loaded",
    "Индикатор загрузки модели (1=загружена, 0=не загружена)"
)

# Устанавливаем gauge в зависимости от состояния модели
if handler.model is not None:
    model_loaded_gauge.set(1)
else:
    model_loaded_gauge.set(0)

# Health check endpoints
@app.get("/")
async def root():
    """Корневой эндпоинт с информацией о сервисе."""
    return {
        "service": "Bank Product Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/api/prediction/"
        }
    }

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса."""
    return {
        "status": "healthy" if handler.model is not None else "degraded",
        "model_loaded": handler.model is not None,
        "model_type": str(type(handler.model)) if handler.model else None
    }

@app.post("/api/prediction/")
async def get_recommendations_for_user(
    user_id: int,
    
    model_params: Dict[str, Any] = Body(
        example={
        "age": 30,
        "antiguedad": 12,
        "renta": 90000,
        "ind_nuevo": 1,
        "indrel": 1,
        "cod_prov": 15.0,
        "ind_actividad_cliente": 1,
        "month": 3
    },
        description="Признаки пользователя для модели"
    )
):
    """Функция для получения рекомендаций банковских продуктов.

    Args:
        user_id (int): Идентификатор пользователя.
        model_params (dict): Параметры пользователя, которые подаются в модель.

    Returns:
        dict: Рекомендации банковских продуктов (топ-5).
    """
    # Формируем параметры для обработчика
    all_params = {
        "user_id": user_id,
        "features": model_params
    }
    
    # Получаем предсказания
    result = handler.handle(all_params)
    
    # Обновляем метрики Prometheus
    main_app_counter.inc()
    main_app_users.inc()
    
    # Если есть рекомендации, обновляем гистограмму вероятностей
    if result.get("status") == "success" and "recommendations" in result:
        recommendations = result["recommendations"]
        if recommendations:
            # Записываем максимальную вероятность в гистограмму
            max_probability = max(rec.get("probability", 0) for rec in recommendations)
            main_app_predictions.observe(max_probability)
    
    return result


@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware для логирования всех запросов."""
    response = await call_next(request)
    
    # Логируем только prediction запросы
    if request.url.path == "/api/prediction/":
        print(f"Request: {request.method} {request.url.path}")
    
    return response

