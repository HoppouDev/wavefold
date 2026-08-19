mod ui;

fn main() -> iced::Result {
    tracing_subscriber::fmt::init();
    iced::application(ui::App::default, ui::App::update, ui::App::view)
        .title("DCT GPU Video Encoder")
        .window_size((640.0, 640.0))
        .run()
}
