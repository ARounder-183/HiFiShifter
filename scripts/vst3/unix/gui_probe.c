/* gui_probe.c —— 实测 GTK3/WebKitGTK 在本机（WSLg）能不能真的画出来。
 *
 * 四件事，任何一件不成立，插件窗口就是"空面板"：
 *   1. GTK3 能不能在某个 GDK 后端上初始化
 *   2. WebKit 的 web process 能不能起来（容器/WSL 里 bubblewrap 常失败）
 *   3. 页面能不能 load-finished
 *   4. GtkPlug 能不能挂在宿主的 X11 窗口里（只有 X11 后端可以）
 */
#include <gtk/gtk.h>
#include <gtk/gtkx.h>
#include <gdk/gdkx.h>
#include <webkit2/webkit2.h>
#include <stdio.h>
#include <string.h>

static gboolean finished = FALSE;
static gboolean failed = FALSE;
static gboolean terminated = FALSE;
static char page_title[256] = "";

static void on_load_changed(WebKitWebView *view, WebKitLoadEvent event, gpointer data) {
    (void)data;
    if (event == WEBKIT_LOAD_FINISHED) {
        finished = TRUE;
        const char *title = webkit_web_view_get_title(view);
        if (title) snprintf(page_title, sizeof(page_title), "%s", title);
        printf("  load-changed: FINISHED title=%s\n", title ? title : "(null)");
    } else {
        printf("  load-changed: %d\n", (int)event);
    }
}

static void on_load_failed(WebKitWebView *view, WebKitLoadEvent event, const char *uri,
                           GError *error, gpointer data) {
    (void)view; (void)event; (void)data;
    failed = TRUE;
    printf("  load-failed: uri=%s error=%s\n", uri ? uri : "?", error ? error->message : "?");
}

static void on_terminated(WebKitWebView *view, WebKitWebProcessTerminationReason reason, gpointer d) {
    (void)view; (void)d;
    terminated = TRUE;
    printf("  web-process-terminated: reason=%d\n", (int)reason);
}

static gboolean quit_loop(gpointer loop) {
    g_main_loop_quit((GMainLoop *)loop);
    return G_SOURCE_REMOVE;
}

static void report_display(void) {
    GdkDisplay *display = gdk_display_get_default();
    if (!display) { printf("  gdk display: NONE\n"); return; }
    printf("  gdk display: %s  name=%s\n",
           G_OBJECT_TYPE_NAME(display), gdk_display_get_name(display));
}

static void run_webview_test(const char *label, GtkWidget *parent) {
    printf("[%s]\n", label);
    GtkWidget *view = webkit_web_view_new();
    g_signal_connect(view, "load-changed", G_CALLBACK(on_load_changed), NULL);
    g_signal_connect(view, "load-failed", G_CALLBACK(on_load_failed), NULL);
    g_signal_connect(view, "web-process-terminated", G_CALLBACK(on_terminated), NULL);
    gtk_container_add(GTK_CONTAINER(parent), view);
    gtk_widget_show_all(parent);

    const char *html =
        "<html><head><title>probe-ok</title></head>"
        "<body style='background:#123'><h1 style='color:#fff'>hi</h1></body></html>";
    webkit_web_view_load_html(WEBKIT_WEB_VIEW(view), html, NULL);

    GMainLoop *loop = g_main_loop_new(NULL, FALSE);
    g_timeout_add(6000, quit_loop, loop);
    g_main_loop_run(loop);
    g_main_loop_unref(loop);

    printf("  => finished=%d failed=%d terminated=%d title=%s\n",
           finished, failed, terminated, page_title[0] ? page_title : "(none)");
    finished = failed = terminated = FALSE;
    page_title[0] = 0;
}

int main(int argc, char **argv) {
    gtk_init(&argc, &argv);
    report_display();

    printf("[1] toplevel window + WebView\n");
    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_default_size(GTK_WINDOW(window), 400, 200);
    run_webview_test("1", window);

    printf("[2] GtkPlug embedded into a real X11 window id\n");
    if (GDK_IS_X11_DISPLAY(gdk_display_get_default())) {
        GtkWidget *host = gtk_window_new(GTK_WINDOW_TOPLEVEL);
        gtk_window_set_default_size(GTK_WINDOW(host), 400, 200);
        gtk_widget_show_all(host);
        while (gtk_events_pending()) gtk_main_iteration();
        GdkWindow *gdk_host = gtk_widget_get_window(host);
        if (gdk_host) {
            GtkWidget *plug = gtk_plug_new(gdk_x11_window_get_xid(gdk_host));
            run_webview_test("2", plug);
            gtk_widget_destroy(plug);
        } else {
            printf("  host window has no GdkWindow\n");
        }
        gtk_widget_destroy(host);
    } else {
        printf("  skipped: the current GDK backend cannot host a GtkPlug\n");
    }
    printf("done\n");
    return 0;
}
