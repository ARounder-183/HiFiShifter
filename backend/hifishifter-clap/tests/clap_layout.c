/* CLAP 1.1.9 ABI oracle: the pinned SDK headers compute the real layout.
 *
 * 【Why a native oracle】Rust's hand-written #[repr(C)] mirrors and the C black-box
 * probe are two independent copies of the same assumption. If BOTH are wrong in the
 * same way, "dlopen works" and "process passes" prove nothing about a real host.
 * Only the SDK header can say where a field actually lives. Kept C99 on purpose:
 * on Linux/macOS the CLAP workflow has cc but not necessarily a C++ toolchain.
 */
#include <stddef.h>
#include <stdio.h>
#include "clap/clap.h"

#define TYPE_LAYOUT(T) printf("\"" #T ".size\":%zu,\"" #T ".align\":%zu,", sizeof(T), _Alignof(T))
#define FIELD_LAYOUT(T, F) printf("\"" #T "." #F "\":%zu,", offsetof(T, F))
/* TYPE_LAYOUT already emits "<T>.size"; a field literally named size would
 * overwrite it in the JSON. Give that one field a distinct key. */
#define FIELD_LAYOUT_NAMED(T, F, KEY) printf("\"" #T "." #KEY "\":%zu,", offsetof(T, F))

int main(void) {
    printf("{");
    TYPE_LAYOUT(clap_version_t);
    TYPE_LAYOUT(clap_event_header_t);
    FIELD_LAYOUT_NAMED(clap_event_header_t, size, size_field);
    FIELD_LAYOUT(clap_event_header_t, time);
    FIELD_LAYOUT(clap_event_header_t, space_id);
    FIELD_LAYOUT(clap_event_header_t, type);
    FIELD_LAYOUT(clap_event_header_t, flags);
    TYPE_LAYOUT(clap_event_transport_t);
    FIELD_LAYOUT(clap_event_transport_t, flags);
    FIELD_LAYOUT(clap_event_transport_t, song_pos_beats);
    FIELD_LAYOUT(clap_event_transport_t, song_pos_seconds);
    FIELD_LAYOUT(clap_event_transport_t, tempo);
    FIELD_LAYOUT(clap_event_transport_t, tempo_inc);
    FIELD_LAYOUT(clap_event_transport_t, loop_start_beats);
    FIELD_LAYOUT(clap_event_transport_t, loop_end_beats);
    FIELD_LAYOUT(clap_event_transport_t, loop_start_seconds);
    FIELD_LAYOUT(clap_event_transport_t, loop_end_seconds);
    FIELD_LAYOUT(clap_event_transport_t, bar_start);
    FIELD_LAYOUT(clap_event_transport_t, bar_number);
    FIELD_LAYOUT(clap_event_transport_t, tsig_num);
    FIELD_LAYOUT(clap_event_transport_t, tsig_denom);
    TYPE_LAYOUT(clap_audio_buffer_t);
    FIELD_LAYOUT(clap_audio_buffer_t, data32);
    FIELD_LAYOUT(clap_audio_buffer_t, data64);
    FIELD_LAYOUT(clap_audio_buffer_t, channel_count);
    FIELD_LAYOUT(clap_audio_buffer_t, latency);
    FIELD_LAYOUT(clap_audio_buffer_t, constant_mask);
    TYPE_LAYOUT(clap_process_t);
    FIELD_LAYOUT(clap_process_t, steady_time);
    FIELD_LAYOUT(clap_process_t, frames_count);
    FIELD_LAYOUT(clap_process_t, transport);
    FIELD_LAYOUT(clap_process_t, audio_inputs);
    FIELD_LAYOUT(clap_process_t, audio_outputs);
    FIELD_LAYOUT(clap_process_t, audio_inputs_count);
    FIELD_LAYOUT(clap_process_t, audio_outputs_count);
    FIELD_LAYOUT(clap_process_t, in_events);
    FIELD_LAYOUT(clap_process_t, out_events);
    TYPE_LAYOUT(clap_plugin_descriptor_t);
    FIELD_LAYOUT(clap_plugin_descriptor_t, clap_version);
    FIELD_LAYOUT(clap_plugin_descriptor_t, id);
    FIELD_LAYOUT(clap_plugin_descriptor_t, name);
    FIELD_LAYOUT(clap_plugin_descriptor_t, vendor);
    FIELD_LAYOUT(clap_plugin_descriptor_t, url);
    FIELD_LAYOUT(clap_plugin_descriptor_t, manual_url);
    FIELD_LAYOUT(clap_plugin_descriptor_t, support_url);
    FIELD_LAYOUT(clap_plugin_descriptor_t, version);
    FIELD_LAYOUT(clap_plugin_descriptor_t, description);
    FIELD_LAYOUT(clap_plugin_descriptor_t, features);
    TYPE_LAYOUT(clap_host_t);
    FIELD_LAYOUT(clap_host_t, clap_version);
    FIELD_LAYOUT(clap_host_t, host_data);
    FIELD_LAYOUT(clap_host_t, name);
    FIELD_LAYOUT(clap_host_t, vendor);
    FIELD_LAYOUT(clap_host_t, url);
    FIELD_LAYOUT(clap_host_t, version);
    FIELD_LAYOUT(clap_host_t, get_extension);
    FIELD_LAYOUT(clap_host_t, request_restart);
    FIELD_LAYOUT(clap_host_t, request_process);
    FIELD_LAYOUT(clap_host_t, request_callback);
    TYPE_LAYOUT(clap_plugin_t);
    FIELD_LAYOUT(clap_plugin_t, desc);
    FIELD_LAYOUT(clap_plugin_t, plugin_data);
    FIELD_LAYOUT(clap_plugin_t, init);
    FIELD_LAYOUT(clap_plugin_t, destroy);
    FIELD_LAYOUT(clap_plugin_t, activate);
    FIELD_LAYOUT(clap_plugin_t, deactivate);
    FIELD_LAYOUT(clap_plugin_t, start_processing);
    FIELD_LAYOUT(clap_plugin_t, stop_processing);
    FIELD_LAYOUT(clap_plugin_t, reset);
    FIELD_LAYOUT(clap_plugin_t, process);
    FIELD_LAYOUT(clap_plugin_t, get_extension);
    FIELD_LAYOUT(clap_plugin_t, on_main_thread);
    TYPE_LAYOUT(clap_plugin_factory_t);
    FIELD_LAYOUT(clap_plugin_factory_t, get_plugin_count);
    FIELD_LAYOUT(clap_plugin_factory_t, get_plugin_descriptor);
    FIELD_LAYOUT(clap_plugin_factory_t, create_plugin);
    TYPE_LAYOUT(clap_plugin_entry_t);
    FIELD_LAYOUT(clap_plugin_entry_t, clap_version);
    FIELD_LAYOUT(clap_plugin_entry_t, init);
    FIELD_LAYOUT(clap_plugin_entry_t, deinit);
    FIELD_LAYOUT(clap_plugin_entry_t, get_factory);
    TYPE_LAYOUT(clap_istream_t);
    FIELD_LAYOUT(clap_istream_t, ctx);
    FIELD_LAYOUT(clap_istream_t, read);
    TYPE_LAYOUT(clap_ostream_t);
    FIELD_LAYOUT(clap_ostream_t, ctx);
    FIELD_LAYOUT(clap_ostream_t, write);
    TYPE_LAYOUT(clap_audio_port_info_t);
    FIELD_LAYOUT(clap_audio_port_info_t, id);
    FIELD_LAYOUT(clap_audio_port_info_t, name);
    FIELD_LAYOUT(clap_audio_port_info_t, flags);
    FIELD_LAYOUT(clap_audio_port_info_t, channel_count);
    FIELD_LAYOUT(clap_audio_port_info_t, port_type);
    FIELD_LAYOUT(clap_audio_port_info_t, in_place_pair);
    TYPE_LAYOUT(clap_plugin_state_t);
    FIELD_LAYOUT(clap_plugin_state_t, save);
    FIELD_LAYOUT(clap_plugin_state_t, load);
    TYPE_LAYOUT(clap_plugin_render_t);
    FIELD_LAYOUT(clap_plugin_render_t, has_hard_realtime_requirement);
    FIELD_LAYOUT(clap_plugin_render_t, set);
    TYPE_LAYOUT(clap_plugin_timer_support_t);
    FIELD_LAYOUT(clap_plugin_timer_support_t, on_timer);
    TYPE_LAYOUT(clap_window_t);
    FIELD_LAYOUT(clap_window_t, api);
    FIELD_LAYOUT(clap_window_t, ptr);
    TYPE_LAYOUT(clap_plugin_gui_t);
    FIELD_LAYOUT(clap_plugin_gui_t, is_api_supported);
    FIELD_LAYOUT(clap_plugin_gui_t, get_preferred_api);
    FIELD_LAYOUT(clap_plugin_gui_t, create);
    FIELD_LAYOUT(clap_plugin_gui_t, destroy);
    FIELD_LAYOUT(clap_plugin_gui_t, set_scale);
    FIELD_LAYOUT(clap_plugin_gui_t, get_size);
    FIELD_LAYOUT(clap_plugin_gui_t, can_resize);
    FIELD_LAYOUT(clap_plugin_gui_t, get_resize_hints);
    FIELD_LAYOUT(clap_plugin_gui_t, adjust_size);
    FIELD_LAYOUT(clap_plugin_gui_t, set_size);
    FIELD_LAYOUT(clap_plugin_gui_t, set_parent);
    FIELD_LAYOUT(clap_plugin_gui_t, set_transient);
    FIELD_LAYOUT(clap_plugin_gui_t, suggest_title);
    FIELD_LAYOUT(clap_plugin_gui_t, show);
    FIELD_LAYOUT(clap_plugin_gui_t, hide);
    TYPE_LAYOUT(clap_host_gui_t);
    FIELD_LAYOUT(clap_host_gui_t, resize_hints_changed);
    FIELD_LAYOUT(clap_host_gui_t, request_resize);
    FIELD_LAYOUT(clap_host_gui_t, request_show);
    FIELD_LAYOUT(clap_host_gui_t, request_hide);
    FIELD_LAYOUT(clap_host_gui_t, closed);
    TYPE_LAYOUT(clap_host_log_t);
    FIELD_LAYOUT(clap_host_log_t, log);
    TYPE_LAYOUT(clap_plugin_latency_t);
    FIELD_LAYOUT(clap_plugin_latency_t, get);
    TYPE_LAYOUT(clap_plugin_tail_t);
    FIELD_LAYOUT(clap_plugin_tail_t, get);
    printf("\"end\":0}\n");
    return 0;
}
