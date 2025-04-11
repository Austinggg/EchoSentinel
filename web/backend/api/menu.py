from flask import Blueprint

from utils.HttpResponse import HttpResponse

bp = Blueprint("menu", __name__)


# TODO
# 暂时先返回这个
@bp.get("/api/menu/all")
def menu_all():
    return HttpResponse.success(
        data=[
            {
                "meta": {"order": -1, "title": "page.dashboard.title"},
                "name": "Dashboard",
                "path": "/dashboard",
                "redirect": "/analytics",
                "children": [
                    {
                        "name": "Analytics",
                        "path": "/analytics",
                        "component": "/dashboard/analytics/index",
                        "meta": {
                            "affixTab": True,
                            "title": "page.dashboard.analytics",
                        },
                    },
                    {
                        "name": "Workspace",
                        "path": "/workspace",
                        "component": "/dashboard/workspace/index",
                        "meta": {"title": "page.dashboard.workspace"},
                    },
                ],
            },
            {
                "meta": {
                    "icon": "ic:baseline-view-in-ar",
                    "keepAlive": True,
                    "order": 1000,
                    "title": "demos.title",
                },
                "name": "Demos",
                "path": "/demos",
                "redirect": "/demos/access",
                "children": [
                    {
                        "name": "AccessDemos",
                        "path": "/demosaccess",
                        "meta": {
                            "icon": "mdi:cloud-key-outline",
                            "title": "demos.access.backendPermissions",
                        },
                        "redirect": "/demos/access/page-control",
                        "children": [
                            {
                                "name": "AccessPageControlDemo",
                                "path": "/demos/access/page-control",
                                "component": "/demos/access/index",
                                "meta": {
                                    "icon": "mdi:page-previous-outline",
                                    "title": "demos.access.pageAccess",
                                },
                            },
                            {
                                "name": "AccessButtonControlDemo",
                                "path": "/demos/access/button-control",
                                "component": "/demos/access/button-control",
                                "meta": {
                                    "icon": "mdi:button-cursor",
                                    "title": "demos.access.buttonControl",
                                },
                            },
                            {
                                "name": "AccessMenuVisible403Demo",
                                "path": "/demos/access/menu-visible-403",
                                "component": "/demos/access/menu-visible-403",
                                "meta": {
                                    "authority": ["no-body"],
                                    "icon": "mdi:button-cursor",
                                    "menuVisibleWithForbidden": True,
                                    "title": "demos.access.menuVisible403",
                                },
                            },
                            {
                                "component": "/demos/access/super-visible",
                                "meta": {
                                    "icon": "mdi:button-cursor",
                                    "title": "demos.access.superVisible",
                                },
                                "name": "AccessSuperVisibleDemo",
                                "path": "/demos/access/super-visible",
                            },
                        ],
                    }
                ],
            },
        ],
    )
