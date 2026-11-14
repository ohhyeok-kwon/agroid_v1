use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tauri::{WebviewUrl, WebviewWindowBuilder};

#[derive(serde::Deserialize)]
struct Startup {
    window: StartupWindow,
}

#[derive(serde::Deserialize)]
struct StartupWindow {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

struct Backend(Option<Child>);

impl Backend {
    fn stop(&mut self) {
        let Some(mut child) = self.0.take() else { return };
        // The Python backend stops rendering, simulation and its observers.
        unsafe { libc::kill(child.id() as i32, libc::SIGTERM) };
        let until = Instant::now() + Duration::from_secs(6);
        while Instant::now() < until {
            if matches!(child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        self.stop();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let startup: Startup = serde_json::from_str(include_str!("../startup.json"))?;
    let root = std::env::var_os("OPERATION_CENTER_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let mut backend = Backend(Some(
        Command::new("python3")
            .arg(root.join("server.py"))
            .args(["--port", "0", "--parent", &std::process::id().to_string()])
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?,
    ));
    let mut url = String::new();
    BufReader::new(backend.0.as_mut().ok_or("backend unavailable")?
        .stdout.take().ok_or("backend stdout unavailable")?)
        .read_line(&mut url)?;
    let url = url.trim().parse().map_err(|_| "backend did not start")?;
    // Wayland does not let an application restore its absolute window position.
    // Use XWayland for this window, including when the desktop uses Wayland.
    std::env::set_var("GDK_BACKEND", "x11");
    tauri::Builder::default()
        .setup(move |app| {
            let window = WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url))
                .title("Operation Center")
                .inner_size(startup.window.width, startup.window.height)
                .min_inner_size(960.0, 640.0)
                .position(startup.window.x, startup.window.y)
                .build()?;
            window.set_position(tauri::LogicalPosition::new(startup.window.x, startup.window.y))?;
            println!("Operation Center 창을 열었습니다.");
            Ok(())
        })
        .build(tauri::generate_context!())?
        .run(move |_, event| {
            if matches!(event, tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit)
                || matches!(event, tauri::RunEvent::WindowEvent {
                    event: tauri::WindowEvent::Destroyed, ..
                })
            {
                backend.stop();
            }
        });
    Ok(())
}
