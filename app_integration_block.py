# ============================================================
# ADD THESE IMPORTS NEAR TOP OF app.py
# ============================================================
from modules.auth import init_users_db, login_user, register_user
from modules.whatsapp_service import normalize_whatsapp_number, send_whatsapp_alert


# ============================================================
# ADD THIS AFTER init_db() OR SESSION INITIALIZATION
# ============================================================
init_users_db()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "auto_alert_logs" not in st.session_state:
    st.session_state.auto_alert_logs = []
if "auto_alert_sent" not in st.session_state:
    st.session_state.auto_alert_sent = 0


# ============================================================
# REPLACE YOUR OLD login_screen() WITH THIS
# ============================================================
def login_screen() -> None:
    st.markdown("""
    <div class="login-card">
        <div class="login-title">✈ AeroLogix AI Login</div>
        <div class="login-sub">Admin = full dashboard. Passenger = flight status only.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        mode = st.radio("Mode", ["Login", "Register"], horizontal=True)
        role = st.radio("Account type", ["Admin", "Passenger"], horizontal=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if mode == "Register":
            if st.button("Create Account", type="primary", use_container_width=True):
                ok, msg = register_user(username, password, role.lower())
                st.success(msg) if ok else st.error(msg)

        else:
            if st.button("Login", type="primary", use_container_width=True):
                ok, msg, user_role = login_user(username, password, role.lower())
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username = username.strip().lower()
                    st.session_state.user_role = user_role
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)


def logout() -> None:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.user_role = None
    st.rerun()


# ============================================================
# ADD AUTH GATE BEFORE SIDEBAR CODE
# ============================================================
if not st.session_state.logged_in:
    login_screen()
    st.stop()

if st.session_state.user_role == "passenger":
    passenger_page()
    st.stop()


# ============================================================
# ADD AI AUTO ALERT FUNCTIONS
# ============================================================
def classify_alert_priority(delay_minutes: float) -> tuple[str, str]:
    if delay_minutes >= 90:
        return "Critical", "Major disruption. Immediate passenger alert required."
    if delay_minutes >= 60:
        return "High", "High delay. Passenger should be updated now."
    if delay_minutes >= 30:
        return "Medium", "Moderate delay. Monitor and notify if needed."
    return "Low", "No urgent alert needed."


def auto_trigger_delay_alerts(
    df_local: pd.DataFrame,
    passenger_number: str,
    delay_threshold: int = 60,
    max_alerts: int = 10,
) -> tuple[int, list[str]]:
    if df_local is None or df_local.empty:
        return 0, ["No dataset loaded."]

    if "Delay_Minutes" not in df_local.columns:
        return 0, ["Delay_Minutes column missing."]

    passenger_number = normalize_whatsapp_number(passenger_number)

    if not passenger_number:
        return 0, ["Passenger WhatsApp number is required."]

    critical = df_local[df_local["Delay_Minutes"] >= delay_threshold].copy()

    if critical.empty:
        return 0, [f"No flights found with delay >= {delay_threshold} minutes."]

    critical = critical.sort_values("Delay_Minutes", ascending=False).head(max_alerts)

    sent_count = 0
    logs = []

    for _, row in critical.iterrows():
        delay = float(row.get("Delay_Minutes", 0) or 0)
        priority, reason = classify_alert_priority(delay)

        message = (
            f"AeroLogix AI Alert\n"
            f"Priority: {priority}\n"
            f"Reason: {reason}\n\n"
            + build_flight_message(row)
        )

        ok, info = send_whatsapp_alert(passenger_number, message)
        flight_id = row.get("Flight_ID", "Unknown")

        if ok:
            sent_count += 1
            logs.append(f"✅ {flight_id} | {priority} | {int(delay)} min | Sent")
        else:
            logs.append(f"❌ {flight_id} | {priority} | {int(delay)} min | {info}")

    return sent_count, logs


def render_auto_alert_panel(df_local: pd.DataFrame) -> None:
    st.markdown("### 🤖 AI Auto Alerts")

    c1, c2, c3 = st.columns(3)
    with c1:
        threshold = st.slider("Delay threshold", 15, 120, 60, 15)
    with c2:
        max_alerts = st.number_input("Max alerts", 1, 50, 10)
    with c3:
        number = st.text_input("Passenger WhatsApp number", placeholder="+94703394005")

    if st.button("🚨 Run AI Auto Alerts", type="primary", use_container_width=True):
        sent, logs = auto_trigger_delay_alerts(df_local, number, threshold, int(max_alerts))
        st.session_state.auto_alert_sent = sent
        st.session_state.auto_alert_logs = logs

    if st.session_state.auto_alert_logs:
        st.success(f"Auto alert completed. Sent: {st.session_state.auto_alert_sent}")
        for log in st.session_state.auto_alert_logs:
            if log.startswith("✅"):
                st.success(log)
            else:
                st.error(log)


# ============================================================
# USE THIS INSIDE ADMIN PASSENGER ALERTS PAGE
# ============================================================
# tab1, tab2 = st.tabs(["Manual Alert", "AI Auto Alerts"])
# with tab1:
#     passenger_alerts_tab(df)
# with tab2:
#     render_auto_alert_panel(df)
