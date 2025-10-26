import React from "react";

export default function Sidebar({ active = "Home" }) {
  const menuItems = [
    { label: "Home", icon: "🏠" },
    { label: "Predictions", icon: "📈" },
    { label: "Team Insights", icon: "👥" },
    { label: "Player Stats", icon: "👤" },
    { label: "Simulations", icon: "📊" },
    { label: "Settings", icon: "⚙️" },
  ];

  return (
    <div
  style={{
    position: "fixed",
    top: 0,
    left: 0,
    width: "250px",
    height: "100vh",
    backgroundColor: "#0D1117",
    color: "#E5E7EB",
    display: "flex",
    flexDirection: "column",
    justifyContent: "space-between",
    padding: "1.5rem 1rem",
    boxSizing: "border-box",
    borderRight: "1px solid #1F2937",
    fontFamily: "'Inter', sans-serif",
  }}
>
      {/* Top Section */}
      <div>
        {/* Logo */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            marginBottom: "2.5rem",
          }}
        >
          <div
            style={{
              width: "14px",
              height: "14px",
              borderRadius: "50%",
              backgroundColor: "#00FF87",
              marginRight: "10px",
            }}
          ></div>
          <div>
            <h2
              style={{
                margin: 0,
                color: "#00FF87",
                fontSize: "1.15rem",
                fontWeight: "600",
                letterSpacing: "0.5px",
              }}
            >
              FutStat
            </h2>
            <p
              style={{
                margin: 0,
                fontSize: "0.8rem",
                color: "#9CA3AF",
              }}
            >
              Premier League Analytics
            </p>
          </div>
        </div>

        {/* Menu */}
        <div>
          {menuItems.map((item) => {
            const isActive = active === item.label;
            return (
              <div
                key={item.label}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "12px",
                  padding: "0.7rem 1rem",
                  marginBottom: "0.5rem",
                  borderRadius: "10px",
                  backgroundColor: isActive ? "#0F3A28" : "transparent",
                  color: isActive ? "#00FF87" : "#D1D5DB",
                  fontWeight: "500",
                  cursor: "pointer",
                  transition: "all 0.2s ease-in-out",
                }}
                onMouseEnter={(e) => {
                  if (!isActive)
                    e.currentTarget.style.backgroundColor = "#1F2937";
                }}
                onMouseLeave={(e) => {
                  if (!isActive)
                    e.currentTarget.style.backgroundColor = "transparent";
                }}
              >
                <span style={{ fontSize: "1rem" }}>{item.icon}</span>
                <span
                  style={{
                    fontSize: "0.95rem",
                    userSelect: "none",
                  }}
                >
                  {item.label}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer */}
      <div
        style={{
          textAlign: "center",
          fontSize: "0.8rem",
          color: "#9CA3AF",
          borderTop: "1px solid #1F2937",
          paddingTop: "1rem",
          marginTop: "2rem",
        }}
      >
        Season 2024/25
      </div>
    </div>
  );
}
