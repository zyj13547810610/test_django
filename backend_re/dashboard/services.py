from ..backend.models import Detection, DetectionStatus, Alert, AlertType, Device, Settings
import os, json, time, logging
from django.utils import timezone
from django.db.models import Count
from django.db.models.functions import TruncDate
from datetime import timedelta
import psutil





logger = logging.getLogger(__name__)
STARTUP_TIME_FILE = "system_startup_time.json"

def get_startup_time():
    try:
        if os.path.exists(STARTUP_TIME_FILE):
            with open(STARTUP_TIME_FILE, 'r') as f:
                return json.load(f).get('startup_time')
        return None
    except Exception as e:
        logger.error(f"读取启动时间文件失败: {e}")
        return None

def save_startup_time(startup_time):
    try:
        with open(STARTUP_TIME_FILE, 'w') as f:
            json.dump({'startup_time': startup_time}, f)
    except Exception as e:
        logger.error(f"保存启动时间文件失败: {e}")

SYSTEM_START_TIME = get_startup_time() or time.time()
if get_startup_time() is None:
    save_startup_time(SYSTEM_START_TIME)


class DashboardService:
    def __init__(self):
        """初始化服务，无需数据库会话"""
        pass
    
    
    def get_system_status(self):
        """获取系统状态信息"""
        try:
            # 获取系统启动时间
            startup_time = SYSTEM_START_TIME

            # 尝试从数据库中读取系统配置
            system_startup_setting = Settings.objects.filter(
                category='system',
                key='startup_time'
            ).first()

            # 如果数据库中有配置，使用数据库中的值
            if system_startup_setting:
                try:
                    startup_time = float(system_startup_setting.value)
                except (ValueError, TypeError):
                    # 如果数据库中的值无效，使用默认值并更新数据库
                    startup_time = SYSTEM_START_TIME
                    system_startup_setting.value = str(startup_time)
                    system_startup_setting.save()
            else:
                # 如果数据库中没有配置，创建新的配置
                try:
                    Settings.objects.create(
                        category='system',
                        key='startup_time',
                        value=str(SYSTEM_START_TIME),
                        description='系统启动时间戳'
                    )
                except Exception as e:
                    logger.error(f"保存系统启动时间到数据库失败: {str(e)}")
            
            # 获取系统运行时间（小时）
            current_time = time.time()
            uptime_hours = round((current_time - startup_time) / 3600, 1)
            
            logger.info(f"系统启动时间: {startup_time}, 当前时间: {current_time}, 运行时间: {uptime_hours}小时")
            
            # 获取CPU和内存使用情况
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            #获取npu显存占用和核心使用率
            npu_info={}
            from ..Get_Npu_Info import SimpleNPUMonitor
             # 获取所有设备信息  
            devices_info = SimpleNPUMonitor().get_all_devices_info()
            for device_id, info in devices_info.items():  
                
                print(f"NPU {device_id}:")  
                print(f"  显存: {info['memory']['used_mb']}/{info['memory']['total_mb']} MB "  
                    f"({info['memory']['usage_percent']}%)")  
                print(f"  AI Core利用率: {info['aicore_utilization_percent']}%")  
                print()
                npu_id=  device_id
                hbm_percent=f"{info['memory']['usage_percent']}%"
                ai_core_percent=f"{info['aicore_utilization_percent']}%"
                npu_info[npu_id]={"hbm_percent":hbm_percent,"ai_core_percent":ai_core_percent}
                pass
                

            # 判断系统状态
            status = "normal"
            message = "dashboard.systemStatus.systemNormal"
            
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                status = "error"
                message = "dashboard.systemStatus.systemResourceStress"
            elif cpu_percent > 70 or memory_percent > 70 or disk_percent > 70:
                status = "warning"
                message = "dashboard.systemStatus.systemLoadHigh"
            
            return {
                "status": status,
                "uptime": uptime_hours,
                "message": message,
                "resources": {
                    "cpu": cpu_percent,
                    "memory": memory_percent,
                    "disk": disk_percent,
                    "npu_info":npu_info
                }
            }
        except Exception as e:
            logger.error(f"获取系统状态时出错: {str(e)}")
            return {
                "status": "error",
                "uptime": 0,
                "message": "dashboard.systemStatus.systemStatusError"
            }
    
    def get_detection_stats(self):
        """获取今日检测统计"""
        try:
            # 获取今天的日期
            today = timezone.now().date()
            yesterday = today - timedelta(days=1)

            logger.info(f"当前系统日期：{today}, 查询检测统计")

            # 查询今日检测总数 - 先检查是否有使用今天日期的记录
            
            today_total = Detection.objects.filter(timestamp__date=today).count() or 0

            logger.info(f"今日记录数：{today_total}")

            # 如果没有今天的记录，查询最近的记录日期
            if today_total == 0:
                # 获取所有检测记录
                all_records = Detection.objects.count() or 0
                logger.info(f"数据库中总记录数：{all_records}")

                # 查找最近的记录
                latest_detection = Detection.objects.order_by('-timestamp').first()

                if latest_detection:
                    # 获取最近记录的日期作为"今天"
                    latest_date = latest_detection.timestamp.date()
                    one_day_before = latest_date - timedelta(days=1)

                    logger.info(f"找到最新记录日期：{latest_date}, 前一天：{one_day_before}")

                    # 使用最近的记录日期重新查询
                    latest_date_total = Detection.objects.filter(timestamp__date=latest_date).count() or 0

                    logger.info(f"使用最新日期 {latest_date} 查询结果：{latest_date_total}")

                    # 直接使用日期字符串比较尝试查询
                    date_str = latest_date.isoformat()
                    manual_count = Detection.objects.filter(timestamp__date=latest_date).count() or 0

                    logger.info(f"使用日期字符串 {date_str} 查询结果：{manual_count}")


                    # 使用此方法查询前一天
                    yesterday_manual_count = Detection.objects.filter(timestamp__date=one_day_before).count() or 0

                    # 使用日期字符串比较的结果
                    if manual_count > 0:
                        change = 0
                        if yesterday_manual_count > 0:
                            change = round(((manual_count - yesterday_manual_count) / yesterday_manual_count) * 100, 1)

                        return {
                            "total": manual_count,
                            "change": change,
                            "today": latest_date.isoformat(),
                            "yesterday_total": yesterday_manual_count,
                            "note": "显示最近的检测记录统计，而非当前日期"
                        }

                    # 如果字符串比较也无法找到记录，返回所有记录总数
                    return {
                        "total": all_records,
                        "change": 0,
                        "today": latest_date.isoformat(),
                        "yesterday_total": 0,
                        "note": "显示所有检测记录统计"
                    }

                # 没有找到任何记录，返回所有记录（应该也是0）
                return {
                    "total": all_records,
                    "change": 0,
                    "today": today.isoformat(),
                    "yesterday_total": 0,
                    "note": "无检测记录"
                }

            # 查询昨日检测总数
            yesterday_total = Detection.objects.filter(timestamp__date=yesterday).count() or 0

            # 计算变化百分比
            change = 0
            if yesterday_total > 0:
                change = round(((today_total - yesterday_total) / yesterday_total) * 100, 1)

            return {
                "total": today_total,
                "change": change,
                "today": today.isoformat(),
                "yesterday_total": yesterday_total
            }
        except Exception as e:
            logger.error(f"获取检测统计时出错: {str(e)}")
            return {
                "total": 0,
                "change": 0
            }

    def get_defect_stats(self):
        """获取缺陷统计"""
        try:
            # 获取今天的日期
            today = timezone.now().date()
            yesterday = today - timedelta(days=1)

            logger.info(f"当前系统日期：{today}, 查询缺陷统计")

            # 获取所有检测记录
            all_records = Detection.objects.count() or 0
            if all_records == 0:
                return {
                    "total": 0,
                    "change": 0,
                    "today": today.isoformat(),
                    "yesterday_total": 0,
                    "note": "无检测记录"
                }

            # 查找最近的记录
            latest_detection = Detection.objects.order_by('-timestamp').first()
            if not latest_detection:
                return {
                    "total": 0,
                    "change": 0,
                    "today": today.isoformat(),
                    "yesterday_total": 0,
                    "note": "无检测记录"
                }

            # 获取最近记录的日期
            latest_date = latest_detection.timestamp.date()
            one_day_before = latest_date - timedelta(days=1)
            logger.info(f"找到最新记录日期：{latest_date}, 前一天：{one_day_before}")

            # 使用日期字符串比较查询
            date_str = latest_date.isoformat()
            defect_count = Detection.objects.filter(timestamp__date=latest_date,status=DetectionStatus.FAIL.value).count() or 0
            logger.info(f"使用日期字符串 {date_str} 查询缺陷结果：{defect_count}")

            # 查询前一天
            yesterday_defect_count = Detection.objects.filter(timestamp__date=one_day_before,status=DetectionStatus.FAIL.value).count() or 0

            # 计算变化百分比
            change = 0
            if yesterday_defect_count > 0:
                change = round(((defect_count - yesterday_defect_count) / yesterday_defect_count) * 100, 1)

            return {
                "total": defect_count,
                "change": change,
                "today": latest_date.isoformat(),
                "yesterday_total": yesterday_defect_count,
                "note": "显示最近的检测记录统计，而非当前日期"
            }

        except Exception as e:
            logger.error(f"获取缺陷统计时出错: {str(e)}")
            return {
                "total": 0,
                "change": 0
            }

    def get_accuracy(self):
        """获取检测准确率统计"""
        try:
            # 获取今天的日期
            today = timezone.now().date()
            yesterday = today - timedelta(days=1)

            logger.info(f"当前系统日期：{today}, 查询准确率统计")

            # 获取所有检测记录
            all_records = Detection.objects.count() or 0
            if all_records == 0:
                return {
                    "value": 0,
                    "change": 0,
                    "today_total": 0,
                    "today_pass": 0,
                    "yesterday_accuracy": 0,
                    "note": "无检测记录"
                }

            # 查找最近的记录
            latest_detection = Detection.objects.order_by('-timestamp').first()
            if not latest_detection:
                return {
                    "value": 0,
                    "change": 0,
                    "today_total": 0,
                    "today_pass": 0,
                    "yesterday_accuracy": 0,
                    "note": "无检测记录"
                }

            # 获取最近记录的日期
            latest_date = latest_detection.timestamp.date()
            one_day_before = latest_date - timedelta(days=1)
            logger.info(f"找到最新记录日期：{latest_date}, 前一天：{one_day_before}")

            # 使用日期字符串比较查询今日总数和通过数
            date_str = latest_date.isoformat()
            today_total = Detection.objects.filter(timestamp__date=latest_date).count() or 0
            
            today_pass = Detection.objects.filter(timestamp__date=latest_date,status=DetectionStatus.PASS).count() or 0

            logger.info(f"使用日期字符串 {date_str} 查询结果：总数={today_total}, 通过={today_pass}")

            # 查询前一天
            yesterday_str = one_day_before.isoformat()
            yesterday_total = Detection.objects.filter(timestamp__date=one_day_before).count() or 0

            yesterday_pass = Detection.objects.filter(timestamp__date=one_day_before,status=DetectionStatus.PASS).count() or 0


            # 计算准确率
            today_accuracy = 0
            if today_total > 0:
                today_accuracy = round((today_pass / today_total) * 100, 1)

            yesterday_accuracy = 0
            if yesterday_total > 0:
                yesterday_accuracy = round((yesterday_pass / yesterday_total) * 100, 1)

            # 计算变化
            change = round(today_accuracy - yesterday_accuracy, 1)

            return {
                "value": today_accuracy,
                "change": change,
                "today_total": today_total,
                "today_pass": today_pass,
                "yesterday_accuracy": yesterday_accuracy,
                "note": "显示最近的检测记录统计，而非当前日期"
            }

        except Exception as e:
            logger.error(f"获取准确率统计时出错: {str(e)}")
            return {
                "value": 0,
                "change": 0
            }

    def get_recent_alerts(self, limit: int = 10):
        """获取最近告警"""
        try:
            alerts = Alert.objects.order_by('-created_at')[:limit]

            result = []
            for alert in alerts:
                device_name = None
                if alert.device:
                    device_name = alert.device.name

                result.append({
                    "id": str(alert.id),
                    "type": alert.type,
                    "message": alert.message,
                    "timestamp": int(alert.created_at.timestamp() * 1000),
                    "device_name": device_name,
                    "is_read": alert.is_read
                })

            return result
        except Exception as e:
            logger.error(f"获取最近告警时出错: {str(e)}")
            return []

    
    def create_alert(self, message: str, alert_type: AlertType = AlertType.INFO, device_id: int = None):
        """创建新的告警"""
        try:
            # 验证设备ID是否存在
            if device_id is not None:
                try:
                    device = Device.objects.get(id=device_id)
                except Device.DoesNotExist:
                    logger.error(f"设备ID {device_id} 不存在")
                    return None
            
            alert = Alert.objects.create(
                type=alert_type,
                message=message,
                device_id=device_id,
                created_at=timezone.now()
            )

            return alert
        except Exception as e:
            logger.error(f"创建告警时出错: {str(e)}")
            return None

    def mark_alert_as_read(self, alert_id: int):
        """将告警标记为已读"""
        try:
            alert = Alert.objects.filter(id=alert_id).first()
            if alert:
                alert.is_read = True
                alert.save()
                return True
            return False
        except Exception as e:
            logger.error(f"标记告警为已读时出错: {str(e)}")
            return False
    
    def get_device_status_summary(self):
        try:
            total = Device.objects.count() or 0
            online = Device.objects.filter(status='online').count() or 0
            offline = Device.objects.filter(status='offline').count() or 0
            error = Device.objects.filter(status='error').count() or 0
            cameras = Device.objects.filter(type='camera').count() or 0

            return {
                "total": total,
                "online": online,
                "offline": offline,
                "error": error,
                "cameras": cameras,
                "online_rate": round((online / total) * 100 if total > 0 else 0, 1)
            }
        except Exception as e:
            logger.error(f"获取设备状态摘要时出错: {str(e)}")
            return {
                "total": 0,
                "online": 0,
                "offline": 0,
                "error": 0,
                "cameras": 0,
                "online_rate": 0
            }

    def get_daily_detection_trends(self, days: int = 7):
        """获取每日检测趋势"""
        try:
            result = []
            today = timezone.now().date()

            logger.info(f"当前系统日期：{today}, 查询检测趋势")

            # 获取所有检测记录
            all_records = Detection.objects.count() or 0
            if all_records == 0:
                # 如果没有记录，返回空数组
                logger.info("没有检测记录，返回空趋势数据")
                return []

            # 查找最近的记录
            latest_detection = Detection.objects.order_by('-timestamp').first()
            if not latest_detection:
                logger.info("没有找到最近的记录，返回空趋势数据")
                return []

            # 获取最近记录的日期
            reference_date = latest_detection.timestamp.date()
            logger.info(f"找到最新记录日期：{reference_date}，计算前{days}天的趋势")

            for i in range(days - 1, -1, -1):
                date = reference_date - timedelta(days=i)
                date_str = date.isoformat()

                # 查询当日检测总数
                total = Detection.objects.filter(timestamp__date=date).count() or 0
                # 查询当日通过数
                pass_count = Detection.objects.filter(timestamp__date=date, status=DetectionStatus.PASS.value).count() or 0
                # 查询当日失败数
                fail_count = Detection.objects.filter(timestamp__date=date, status=DetectionStatus.FAIL.value).count() or 0

                result.append({
                    "date": date.isoformat(),
                    "total": total,
                    "pass": pass_count,
                    "fail": fail_count
                })

                logger.info(f"日期 {date_str} 统计：总数={total}, 通过={pass_count}, 失败={fail_count}")

            return result
        except Exception as e:
            logger.error(f"获取每日检测趋势时出错: {str(e)}")
            return []
