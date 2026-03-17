if source_ms > 0 && target_ms > 0 {
                if let Err(e) = check(unsafe {
                    VslibAddTimeCtrlPnt(proj.0, item_num, source_ms, target_ms)
                }) {
                    eprintln!(
                        "[vslib] WARNING: 
