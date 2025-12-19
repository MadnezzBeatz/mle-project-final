import requests
import time

def test_request(user_id: int) -> None:
    for i in range(3):        
        params = {
            "age": 30,
            "antiguedad": 12,
            "renta": 90000,
            "ind_nuevo": 1,
            "indrel": 1,
            "cod_prov": 15,
            "ind_actividad_cliente": 1,
            "month": 3
            }
        req = requests.post(f"http://localhost:1702/api/prediction/?user_id={user_id+i}", json=params)
        time.sleep(10)
        print(req.status_code)
test_request(1)
