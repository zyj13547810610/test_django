from django.utils import timezone
from rest_framework.exceptions import APIException
from ..backend.models import Settings, Device, AlertType
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class SettingsService:
    def __init__(self):
        """初始化服务，无需数据库会话"""
        pass

    # --- General Settings Methods ---

    def get_settings(self, category: Optional[str] = None) -> List[Settings]:
        """获取指定类别或所有设置"""
        try:
            queryset = Settings.objects.all()
            if category:
                queryset = queryset.filter(category=category)
            return list(queryset)
        except Exception as e:
            logger.error(f"获取设置时出错: {str(e)}")
            return []

    def get_setting(self, category: str, key: str) -> Optional[Settings]:
        """获取单个设置"""
        try:
            return Settings.objects.filter(category=category, key=key).first()
        except Exception as e:
            logger.error(f"获取设置 {category}:{key} 时出错: {str(e)}")
            return None

    def upsert_setting(self, category: str, key: str, value: str, description: Optional[str] = None) -> Settings:
        """创建或更新设置"""
        try:
            setting = self.get_setting(category, key)
            if setting:
                # 更新现有设置
                setting.value = value
                if description:
                    setting.description = description
                setting.updated_at = timezone.now()
                setting.save()
            else:
                # 创建新设置
                setting = Settings.objects.create(
                    category=category,
                    key=key,
                    value=value,
                    description=description or "",
                    created_at=timezone.now(),
                    updated_at=timezone.now()
                )
            return setting
        except Exception as e:
            logger.error(f"创建或更新设置 {category}:{key} 时出错: {str(e)}")
            raise APIException(detail=f"创建或更新设置失败: {str(e)}", code=500)

    def delete_setting(self, category: str, key: str) -> bool:
        """删除设置"""
        try:
            setting = self.get_setting(category, key)
            if not setting:
                return False
            setting.delete()
            return True
        except Exception as e:
            logger.error(f"删除设置 {category}:{key} 时出错: {str(e)}")
            raise APIException(detail=f"删除设置失败: {str(e)}", code=500)

    # --- System Settings Methods ---

    def get_system_settings(self) -> Dict[str, Any]:
        """获取所有系统设置，如果不存在则返回默认值"""
        try:
            settings = self.get_settings("system")
            # 默认设置
            defaults = {
                "auto_save_interval": 5,
                "data_retention": "30天",
                "alarm_threshold": 90,
                "language": "zh-CN"
            }
            result = defaults.copy()

            # 用数据库中的设置覆盖默认值
            for setting in settings:
                if setting.key in ["auto_save_interval", "alarm_threshold"]:
                    try:
                        result[setting.key] = int(setting.value)
                    except (ValueError, TypeError):
                        logger.warning(f"设置 {setting.key} 的值 {setting.value} 无法转换为整数，使用默认值")
                        pass
                else:
                    result[setting.key] = setting.value

            return result
        except Exception as e:
            logger.error(f"获取系统设置时出错: {str(e)}")
            return defaults

    def update_system_settings(self, settings_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新系统设置，确保所有设置都被保存到数据库"""
        try:
            # 获取当前设置
            current_settings = self.get_system_settings()

            # 更新设置
            for key, value in settings_data.items():
                if value is not None:  # 只更新非空值
                    self.upsert_setting("system", key, str(value))

            # 确保所有默认设置都存在于数据库中
            for key, value in current_settings.items():
                if key not in settings_data:
                    self.upsert_setting("system", key, str(value))

            return self.get_system_settings()
        except Exception as e:
            logger.error(f"更新系统设置时出错: {str(e)}")
            raise APIException(detail=f"更新系统设置失败: {str(e)}", code=500)

    # --- MES Integration Settings Methods ---

    def get_mes_settings(self) -> Dict[str, str]:
        """获取MES集成设置"""
        try:
            settings = self.get_settings("mes")
            result = {setting.key: setting.value for setting in settings}
            return result
        except Exception as e:
            logger.error(f"获取MES设置时出错: {str(e)}")
            return {}

    def update_mes_settings(self, settings_data: Dict[str, str]) -> Dict[str, str]:
        """更新MES集成设置"""
        try:
            for key, value in settings_data.items():
                self.upsert_setting("mes", key, value)
            return self.get_mes_settings()
        except Exception as e:
            logger.error(f"更新MES设置时出错: {str(e)}")
            raise APIException(detail=f"更新MES设置失败: {str(e)}", code=500)

    # --- Device Methods ---

    def get_devices(self, type: Optional[str] = None) -> List[Device]:
        """获取设备列表，可选择按类型过滤"""
        try:
            queryset = Device.objects.all()
            if type:
                queryset = queryset.filter(type=type)
            return list(queryset)
        except Exception as e:
            logger.error(f"获取设备列表时出错: {str(e)}")
            return []

    def get_device(self, device_id: int) -> Optional[Device]:
        """获取单个设备"""
        try:
            return Device.objects.filter(id=device_id).first()
        except Exception as e:
            logger.error(f"获取设备 {device_id} 时出错: {str(e)}")
            return None

    def get_device_by_name(self, name: str) -> Optional[Device]:
        """通过名称获取设备"""
        try:
            return Device.objects.filter(name=name).first()
        except Exception as e:
            logger.error(f"通过名称获取设备 {name} 时出错: {str(e)}")
            return None

    def create_device(self, name: str, type: str, model: Optional[str] = None,
                     config: Dict[str, Any] = None, status: str = "offline") -> Device:
        """创建新设备"""
        if config is None:
            config = {}

        try:
            device = Device.objects.create(
                name=name,  
                type=type,
                model=model,
                config=config,
                status=status,
                created_at=timezone.now(),
                updated_at=timezone.now()
            )
            device.save()
            return device
        except Exception as e:
            logger.error(f"创建设备 {name} 时出错: {str(e)}")
            raise APIException(detail=f"创建设备失败: {str(e)}", code=500)

    def update_device(self, device_id: int, data: Dict[str, Any]) -> Device:
        """更新设备"""
        device = self.get_device(device_id)
        if not device:
            raise APIException(detail="设备未找到", code=404)

        # 如果更新名称，检查是否重复
        if 'name' in data and data['name'] != device.name:
            existing_device = self.get_device_by_name(data['name'])
            if existing_device:
                raise APIException(detail=f"设备名称 '{data['name']}' 已存在", code=400)

        try:
            # 更新字段
            for key, value in data.items():
                if key in ['name', 'type', 'model', 'status']:
                    setattr(device, key, value)

            # 特殊处理 config，合并而不是替换
            if 'config' in data:
                if device.config is None:
                    device.config = data['config']
                else:
                    device.config.update(data['config'])

            device.updated_at = timezone.now()
            device.save()
            return device
        except Exception as e:
            logger.error(f"更新设备 {device_id} 时出错: {str(e)}")
            raise APIException(detail=f"更新设备失败: {str(e)}", code=500)

    def delete_device(self, device_id: int) -> bool:
        """删除设备"""
        try:
            device = self.get_device(device_id)
            if not device:
                return False
            device.delete()
            return True
        except Exception as e:
            logger.error(f"删除设备 {device_id} 时出错: {str(e)}")
            raise APIException(detail=f"删除设备失败: {str(e)}", code=500)

    def update_device_status(self, device_id: int, status: str) -> Device:
        """更新设备状态"""
        device = self.get_device(device_id)
        if not device:
            raise APIException(detail="设备未找到", code=404)

        if status not in ["online", "offline", "error"]:
            raise APIException(detail="无效的状态", code=400)

        try:
            device.status = status
            device.updated_at = timezone.now()
            device.save()
            return device
        except Exception as e:
            logger.error(f"更新设备 {device_id} 状态时出错: {str(e)}")
            raise APIException(detail=f"更新设备状态失败: {str(e)}", code=500)