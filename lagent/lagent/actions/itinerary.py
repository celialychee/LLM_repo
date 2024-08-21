import json
import os
import requests
from typing import Optional, Type

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode

class ItineraryPlan(BaseAction):
    """Itinerary plugin for travel planning."""
    
    def __init__(self,
                 key: Optional[str] = None,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True) -> None:
        super().__init__(description, parser, enable)
        key = os.environ.get('ITINERARY_API_KEY', key)
        if key is None:
            raise ValueError(
                'Please set Itinerary API key either in the environment '
                'as ITINERARY_API_KEY or pass it as `key`')
        self.key = key
        # self.itinerary_plan_url_v1 = 'https://restapi.amap.com/v3/direction/walking?'
        self.poi_url = "https://restapi.amap.com/v3/place/text"
        self.walking_url = "https://restapi.amap.com/v3/direction/walking"
        self.bus_url = "https://restapi.amap.com/v3/direction/transit/integrated"
        self.driving_url = "https://restapi.amap.com/v3/direction/driving"
        self.riding_url = "https://restapi.amap.com/v4/direction/bicycling"

    @tool_api
    def location(self, keywords: str, city: str):
        """一个旅行目的地经纬度查询API。可以根据目的地名查询经纬度信息。
        
        Args:
            keywords (:class:`str`): The name to query.
            city (:class:`str`): The city name.
        """
        tool_return = ActionReturn(type=self.name)
        try:
            poi_response = requests.get(
                self.poi_url,
                params={
                    # 请求服务权限标识
                    'key': self.key, \
                    # 查询关键字
                    'keywords': keywords, \
                    # 查询 POI 类型
                    'types': '风景名胜', \
                    # 查询城市
                    'city': city, \
                    'output': "json"
                    }
            )
        except Exception as e:
            return -1, str(e)
        poi_response = poi_response.json()
        # print(poi_response)
        if poi_response['status'] != '1':
            # 返回状态码和状态说明
            # print(poi_response.info)
            tool_return.errmsg = poi_response['info']
            tool_return.state = ActionStatusCode.HTTP_ERROR
            # return poi_response.status, poi_response.info
        else:
            # poi_response = poi_response.json()
            if len(poi_response['pois']) == 0:
                tool_return.errmsg = "未查到该景点"
                tool_return.state = ActionStatusCode.HTTP_ERROR
            else:
                x, y = eval(poi_response['pois'][0]['location'])
                # print(f'{x} {y}')
                data = [
                    f'longitude: {y}', \
                    f'latitude: {x}',
                    f'city: {city}'
                ]
                parsed_res = '\n'.join(data)
                tool_return.result = [dict(type='text', content=str(parsed_res))]
                tool_return.state = ActionStatusCode.SUCCESS
        return tool_return
    
    @tool_api
    def get_walking_route(self, src_lon: float, src_lat: float, des_lon: float, des_lat: float):
        """一个旅行规划查询API。可以根据出发地、目的地名以及出行偏好查询经纬度信息。
        
        Args:
            src_lon (:class:`float`): the origin longitude.
            src_lat (:class:`float`): the origin latitude.
            src_lon (:class:`float`): the destination longitude.
            src_lat (:class:`float`): the destination latitude.
        """
        tool_return = ActionReturn(type=self.name)
        # src_lon = src["longitude"]
        # src_lat = src["latitude"]
        # des_lon = des["longitude"]
        # des_lat = des["latitude"]
        try:
            route_response = requests.get(
                self.walking_url,
                params={
                    # 请求服务权限标识
                    'key': self.key, \
                    # 出发点
                    'origin': f'{src_lon},{src_lat}', \
                    # 目的地
                    'destination': f'{des_lon},{des_lat}', \
                }
            )
        except Exception as e:
            return -1, str(e)
        route_response = route_response.json()
        # print(route_response)
        if route_response['status'] != '1':
            # 返回状态码和状态说明
            tool_return.errmsg = route_response['info']
            tool_return.state = ActionStatusCode.HTTP_ERROR
        else:
            # route_response = route_response.json()
            paths = route_response['route']["paths"]
            if len(paths) == 0:
                tool_return.errmsg = "未查到步行信息"
                tool_return.state = ActionStatusCode.HTTP_ERROR
            else:
                distance = paths[0]["distance"]
                duration = paths[0]["duration"]
                data = [
                    f'distance: {distance}', \
                    f'duration: {duration}'
                ]
                parsed_res = '\n'.join(data)
                tool_return.result = [dict(type='text', content=str(parsed_res))]
                # tool_return.result = dict(distance=distance, duration=duration)
                tool_return.state = ActionStatusCode.SUCCESS
        return tool_return
    
    @tool_api
    def get_bus_route(self, src_lon: float, src_lat: float, src_city: str, des_lon: float, des_lat: float, des_city: str):
        """一个旅行规划查询API。可以根据出发地、目的地名以及出行偏好查询经纬度信息。
        
        Args:
            src_lon (:class:`double`): the origin longitude.
            src_lat (:class:`double`): the origin latitude.
            src_city (:class:`str`): The origin name.
            src_lon (:class:`double`): the destination longitude.
            src_lat (:class:`double`): the destination latitude.
            des_city (:class:`str`): The destination name.
        """
        tool_return = ActionReturn(type=self.name)
        try:
            route_response = requests.get(
                self.bus_url,
                params={
                    # 请求服务权限标识
                    'key': self.key, \
                    # 出发点
                    'origin': f'{src_lon},{src_lat}', \
                    # 目的地
                    'destination': f'{des_lon},{des_lat}', \
                    # 起点城市
                    "city": src_city, \
                    # 目的城市
                    "cityd": des_city
                    }
            )
        except Exception as e:
            return -1, str(e)
        route_response = route_response.json()
        print(route_response)
        if route_response['status'] != '1':
            # 返回状态码和状态说明
            tool_return.errmsg = route_response['info']
            tool_return.state = ActionStatusCode.HTTP_ERROR
        else:
            # route_response = route_response.json()
            transits = route_response['route']["transits"]
            if len(transits) == 0:
                tool_return.errmsg = "未查到公交线路"
                tool_return.state = ActionStatusCode.HTTP_ERROR
            else:
                transit = transits[0]
                cost = transit["cost"]
                duration = transit["duration"]
                segments = transit["segments"]
                parsed_data = []
                for segment in segments:
                    if len(segment["bus"]["buslines"])!=0:
                        data = [
                            f'起始站: {segment["bus"]["buslines"][0]["departure_stop"]["name"]}',
                            f'到达站: {segment["bus"]["buslines"][0]["arrival_stop"]["name"]}',
                            f'线路名称: {segment["bus"]["buslines"][0]["name"]}',
                            f'站数: {segment["bus"]["buslines"][0]["via_num"]}站',
                            f'距离: {segment["bus"]["buslines"][0]["distance"]}m',
                            f'时长: {segment["bus"]["buslines"][0]["duration"]}s',
                        ]
                        parsed_data.append('\n'.join(data))
                parsed_res = '\n\n'.join(parsed_data)
                tool_return.result = [dict(type='text', content=str(parsed_res))]
                tool_return.state = ActionStatusCode.SUCCESS
        return tool_return

# # test
# ip = ItineraryPlan("api")
# lo = ip.location("夫子庙", "南京")
# print(lo)

