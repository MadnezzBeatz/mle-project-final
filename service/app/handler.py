import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

class FastApiHandler:
    """Обработчик запросов для рекомендательной системы банковских продуктов."""
    
    def __init__(self):
        self.param_types = {
            "user_id": int,
            "features": dict
        }
        
        self.model_path = "./models/best_model.bin"

        self.required_features = [
            'age',
            'antiguedad',
            'renta',
            'ind_nuevo',
            'indrel',
            'cod_prov',
            'ind_actividad_cliente',
            'month'
        ]
        
        self.product_names = [
            'ind_ahor_fin_ult1',
            'ind_aval_fin_ult1',
            'ind_cco_fin_ult1',
            'ind_cder_fin_ult1',
            'ind_cno_fin_ult1',
            'ind_ctju_fin_ult1',
            'ind_ctma_fin_ult1',
            'ind_ctop_fin_ult1',
            'ind_ctpp_fin_ult1',
            'ind_deco_fin_ult1',
            'ind_deme_fin_ult1',
            'ind_dela_fin_ult1',
            'ind_ecue_fin_ult1',
            'ind_fond_fin_ult1',
            'ind_hip_fin_ult1',
            'ind_plan_fin_ult1',
            'ind_pres_fin_ult1',
            'ind_reca_fin_ult1',
            'ind_tjcr_fin_ult1',
            'ind_valo_fin_ult1',
            'ind_viv_fin_ult1',
            'ind_nomina_ult1',
            'ind_nom_pens_ult1',
            'ind_recibo_ult1'
        ]
        
        self.load_model()
    
    def load_model(self):
        """Загружаем обученную Random Forest модель."""
        try:
            self.model = joblib.load(self.model_path)
            print(f"Модель загружена из {self.model_path}")
            
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            self.model = None
    
    def _preprocess_features(self, features: Dict[str, Any]) -> pd.DataFrame:
        """Преобразуем входные признаки в формат для модели."""
        # Создаем DataFrame с одним рядом
        df = pd.DataFrame([features])
        
        # Преобразуем категориальные признаки
        if 'sexo' in df.columns:
            df['sexo'] = df['sexo'].map({'H': 1, 'V': 0, 'M': 1}).fillna(1)
        
        if 'ind_empleado' in df.columns:
            # One-hot encoding для статуса занятости
            for emp_status in ['A', 'B', 'F', 'N', 'S']:
                df[f'ind_empleado_{emp_status}'] = (df['ind_empleado'] == emp_status).astype(int)
        
        # Заполняем пропуски
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def get_recommendations(self, user_features: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """Получаем топ-K рекомендованных продуктов."""
        if self.model is None:
            raise ValueError("Модель не загружена!")
        
        # Препроцессинг признаков
        features_df = self._preprocess_features(user_features)
        
        # df = pd.DataFrame([features_df])
        proba = self.model.predict_proba(features_df)
        prob_class1 = [p[0][1] for p in proba]
        top5_indices = np.argsort(prob_class1)[-5:][::-1]

        
        top5_recommendations = []
        for idx in top5_indices:
            top5_recommendations.append({
                'product_id': int(idx),
                'probability': float(prob_class1[idx])
            })
        
        return top5_recommendations
    
    def validate_features(self, features: Dict[str, Any]) -> bool:
        """Проверяем наличие необходимых признаков."""
        # Базовая проверка - есть ли хотя бы некоторые признаки
        if not features:
            return False
        
        # Проверяем числовые значения
        for key, value in features.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    return False
        
        return True
    
    def handle(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Основной метод обработки запроса."""
        try:
            # Проверяем обязательные поля
            if 'user_id' not in params:
                return {'error': 'Отсутствует user_id', 'status': 'error'}
            
            if 'features' not in params:
                return {'error': 'Отсутствуют features', 'status': 'error'}
            
            user_id = params['user_id']
            features = params['features']
            
            # Валидация признаков
            if not self.validate_features(features):
                return {'error': 'Некорректные features', 'status': 'error'}
            
            # Получаем рекомендации
            recommendations = self.get_recommendations(features, top_k=5)
            
            # Формируем ответ
            response = {
                'user_id': int(user_id),
                'recommendations': recommendations,
                'status': 'success'
            }
            
            print(f"✅ Рекомендации для user_id={user_id}")
            
        except Exception as e:
            print(f"❌ Ошибка обработки запроса: {e}")
            response = {
                'error': f'Внутренняя ошибка сервера: {str(e)}',
                'status': 'error'
            }
        
        return response
    
# model_path = "./models/best_model.bin"
# model = joblib.load(model_path)
model_params = {
        "age": 30,
        "antiguedad": 12,
        "renta": 90000,
        "ind_nuevo": 1,
        "indrel": 1,
        "cod_prov": 15.0,
        "ind_actividad_cliente": 1,
        "month": 3
    }
# df = pd.DataFrame([features])
# proba = model.predict_proba(df)
# prob_class1 = [p[0][1] for p in proba]
# top5_indices = np.argsort(prob_class1)[-5:][::-1]
# product_names = [
#     'ind_ahor_fin_ult1',      # 0
#     'ind_aval_fin_ult1',      # 1
#     'ind_cco_fin_ult1',       # 2
#     'ind_cder_fin_ult1',      # 3
#     'ind_cno_fin_ult1',       # 4
#     'ind_ctju_fin_ult1',      # 5
#     'ind_ctma_fin_ult1',      # 6
#     'ind_ctop_fin_ult1',      # 7
#     'ind_ctpp_fin_ult1',      # 8
#     'ind_deco_fin_ult1',      # 9
#     'ind_deme_fin_ult1',      # 10
#     'ind_dela_fin_ult1',      # 11
#     'ind_ecue_fin_ult1',      # 12
#     'ind_fond_fin_ult1',      # 13
#     'ind_hip_fin_ult1',       # 14
#     'ind_plan_fin_ult1',      # 15
#     'ind_pres_fin_ult1',      # 16
#     'ind_reca_fin_ult1',      # 17
#     'ind_tjcr_fin_ult1'       # 18
# ]

# # Формируем топ-5 рекомендаций
# top5_recommendations = []
# for idx in top5_indices:
#     top5_recommendations.append({
#         'product_id': int(idx),
#         'probability': float(prob_class1[idx])
#     })
# print("Топ-5 рекомендованных продуктов:")
# for rec in top5_recommendations:
#     print(f"  {rec['product_id']}: {rec['probability']:.4f}")
