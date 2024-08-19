import json
import os
import requests
from typing import Optional, Type

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode

class WeatherQuery(BaseAction):
    """Weather plugin for querying weather information."""
    
    def __init__(self,
                 key: Optional[str] = None,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True) -> None:
        super().__init__(description, parser, enable)
        key = os.environ.get('WEATHER_API_KEY', key)
        if key is None:
            raise ValueError(
                'Please set Weather API key either in the environment '
                'as WEATHER_API_KEY or pass it as `key`')
        self.key = key
        self.location_query_url = 'https://geoapi.qweather.com/v2/city/lookup'
        # 根据天数不同选择URL
        self.weather_urls = {
            1: 'https://devapi.qweather.com/v7/weather/now',
            3: 'https://devapi.qweather.com/v7/weather/3d',
            7: 'https://devapi.qweather.com/v7/weather/7d',
        }


    @tool_api
    def run(self, query: str, days: int) -> ActionReturn:
        """一个天气查询API。可以根据城市名查询天气信息。
        
        Args:
            query (:class:`str`): The city name to query.
            days (:class:`int`): The number of days to query (1, 3, 7, or 10).
        """
        tool_return = ActionReturn(type=self.name)
        if days not in self.weather_urls:
            tool_return.errmsg = f"Unsupported forecast duration: {days} days."
            tool_return.state = ActionStatusCode.API_ERROR
            return tool_return

        status_code, response = self._search(query, days)
        if status_code == -1:
            tool_return.errmsg = response
            tool_return.state = ActionStatusCode.HTTP_ERROR
        elif status_code == 200:
            parsed_res = self._parse_results(response, days)
            tool_return.result = [dict(type='text', content=str(parsed_res))]
            tool_return.state = ActionStatusCode.SUCCESS
        else:
            tool_return.errmsg = str(status_code)
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return
    

    def _parse_results(self, results: dict, days: int) -> str:
        """Parse the weather results from QWeather API based on forecast days.
        
        Args:
            results (dict): The weather content from QWeather API in json format.
            days (int): The number of forecast days.
        
        Returns:
            str: The parsed weather results.
        """
        if days == 1:
            # 处理当前天气信息
            now = results['now']
            data = [
                f'数据观测时间: {now["obsTime"]}',
                f'温度: {now["temp"]}°C',
                f'体感温度: {now["feelsLike"]}°C',
                f'天气: {now["text"]}',
                f'风向: {now["windDir"]}，角度为 {now["wind360"]}°',
                f'风力等级: {now["windScale"]}，风速为 {now["windSpeed"]} km/h',
                f'相对湿度: {now["humidity"]}',
                f'当前小时累计降水量: {now["precip"]} mm',
                f'大气压强: {now["pressure"]} 百帕',
                f'能见度: {now["vis"]} km',
            ]
            return '\n'.join(data)
        else:
            # 处理多天预报信息
            print(results)
            forecasts = results['daily']
            print(forecasts)
            parsed_data = []
            for forecast in forecasts:
                data = [
                    f'日期: {forecast["fxDate"]}',
                    f'白天天气: {forecast["textDay"]}',
                    f'夜晚天气: {forecast["textNight"]}',
                    f'最高温度: {forecast["tempMax"]}°C',
                    f'最低温度: {forecast["tempMin"]}°C',
                    f'风向: {forecast["windDirDay"]}',
                    f'风力等级: {forecast["windScaleDay"]}',
                ]
                parsed_data.append('\n'.join(data))
            return '\n\n'.join(parsed_data)

    def _search(self, query: str, days: int):
        # get city_code
        try:
            city_code_response = requests.get(
                self.location_query_url,
                params={'key': self.key, 'location': query}
            )
        except Exception as e:
            return -1, str(e)
        if city_code_response.status_code != 200:
            return city_code_response.status_code, city_code_response.json()
        city_code_response = city_code_response.json()
        if len(city_code_response['location']) == 0:
            return -1, '未查询到城市'
        city_code = city_code_response['location'][0]['id']
        # get weather
        try:
            # print("TEST")
            weather_response = requests.get(
                self.weather_urls[days],
                params={'key': self.key, 'location': city_code}
            )
        except Exception as e:
            return -1, str(e)
        return weather_response.status_code, weather_response.json()
