(function setupTranslation() {
  const dictionary = {
    en: {
      app_title: "Plant Disease Detection",
      history: "History",
      upload_leaf: "Upload Leaf Image",
      drag_drop: "Drag & drop image or click to browse",
      original_preview: "Original Preview",
      enhanced_preview: "Enhanced Preview",
      enhancement_module: "Enhancement Module",
      ai_enhancement: "AI Enhancement",
      manual_enhancement: "Manual Enhancement",
      run_ai_enhance: "Run AI Enhance",
      apply_manual: "Apply Manual",
      reset: "Reset",
      analyze: "Analyze",
      predict_offline: "Predict Offline",
      cache_models: "Cache TFJS Models",
      plant: "Plant",
      disease: "Disease",
      severity: "Severity",
      confidence: "Confidence",
      toggle_gradcam: "Toggle Grad-CAM",
      top5: "Top 5 Predictions",
      original: "Original",
      enhanced: "Enhanced",
      cure_plan: "7-Day Cure Timeline",
      feedback: "Feedback",
    },
    te: {
      app_title: "మొక్క వ్యాధి గుర్తింపు",
      history: "చరిత్ర",
      upload_leaf: "ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి",
      drag_drop: "చిత్రాన్ని డ్రాగ్ చేసి వదలండి లేదా బ్రౌజ్ చేయండి",
      original_preview: "అసలు ప్రివ్యూ",
      enhanced_preview: "మెరుగుపరచిన ప్రివ్యూ",
      enhancement_module: "ఎన్‌హాన్స్‌మెంట్ మాడ్యూల్",
      ai_enhancement: "AI ఎన్‌హాన్స్‌మెంట్",
      manual_enhancement: "మాన్యువల్ ఎన్‌హాన్స్‌మెంట్",
      run_ai_enhance: "AI ఎన్‌హాన్స్ నడపండి",
      apply_manual: "మాన్యువల్ వర్తించు",
      reset: "రీసెట్",
      analyze: "విశ్లేషించు",
      predict_offline: "ఆఫ్‌లైన్ అంచనా",
      cache_models: "TFJS మోడళ్లను క్యాష్ చేయండి",
      plant: "మొక్క",
      disease: "వ్యాధి",
      severity: "తీవ్రత",
      confidence: "నమ్మకం",
      toggle_gradcam: "Grad-CAM చూపు/దాచు",
      top5: "టాప్ 5 అంచనాలు",
      original: "అసలు",
      enhanced: "మెరుగుపరచినది",
      cure_plan: "7-రోజుల చికిత్స టైమ్‌లైన్",
      feedback: "అభిప్రాయం",
    },
    hi: {
      app_title: "पौधा रोग पहचान",
      history: "इतिहास",
      upload_leaf: "पत्ती की छवि अपलोड करें",
      drag_drop: "छवि ड्रैग-ड्रॉप करें या ब्राउज़ करें",
      original_preview: "मूल प्रीव्यू",
      enhanced_preview: "एन्हांस्ड प्रीव्यू",
      enhancement_module: "एन्हांसमेंट मॉड्यूल",
      ai_enhancement: "AI एन्हांसमेंट",
      manual_enhancement: "मैनुअल एन्हांसमेंट",
      run_ai_enhance: "AI एन्हांस चलाएं",
      apply_manual: "मैनुअल लागू करें",
      reset: "रीसेट",
      analyze: "विश्लेषण करें",
      predict_offline: "ऑफलाइन भविष्यवाणी",
      cache_models: "TFJS मॉडल कैश करें",
      plant: "पौधा",
      disease: "रोग",
      severity: "गंभीरता",
      confidence: "विश्वसनीयता",
      toggle_gradcam: "Grad-CAM दिखाएं/छिपाएं",
      top5: "टॉप 5 भविष्यवाणियां",
      original: "मूल",
      enhanced: "एन्हांस्ड",
      cure_plan: "7-दिन इलाज टाइमलाइन",
      feedback: "फीडबैक",
    },
    ta: {
      app_title: "தாவர நோய் கண்டறிதல்",
      history: "வரலாறு",
      upload_leaf: "இலை படத்தை பதிவேற்றவும்",
      drag_drop: "படத்தை இழுத்து விடவும் அல்லது தேர்வு செய்யவும்",
      original_preview: "அசல் முன்னோட்டம்",
      enhanced_preview: "மேம்படுத்தப்பட்ட முன்னோட்டம்",
      enhancement_module: "மேம்பாட்டு தொகுதி",
      ai_enhancement: "AI மேம்பாடு",
      manual_enhancement: "கையேடு மேம்பாடு",
      run_ai_enhance: "AI மேம்பாடு இயக்கவும்",
      apply_manual: "கையேடு செயல்",
      reset: "மீட்டமை",
      analyze: "பகுப்பாய்வு",
      predict_offline: "ஆஃப்லைன் கணிப்பு",
      cache_models: "TFJS மாதிரிகளை சேமிக்கவும்",
      plant: "தாவரம்",
      disease: "நோய்",
      severity: "தீவிரம்",
      confidence: "நம்பிக்கை",
      toggle_gradcam: "Grad-CAM காட்ட/மறைக்க",
      top5: "முதல் 5 கணிப்புகள்",
      original: "அசல்",
      enhanced: "மேம்பட்டது",
      cure_plan: "7 நாள் சிகிச்சை காலவரிசை",
      feedback: "கருத்து",
    },
  };

  function applyTranslations(lang) {
    const map = dictionary[lang] || dictionary.en;
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const key = el.getAttribute("data-i18n");
      if (map[key]) el.textContent = map[key];
    });
  }

  window.TranslationManager = {
    init() {
      const switcher = document.getElementById("languageSwitcher");
      const saved = localStorage.getItem("lang") || "en";
      if (switcher) switcher.value = saved;
      applyTranslations(saved);
      switcher?.addEventListener("change", () => {
        const lang = switcher.value;
        localStorage.setItem("lang", lang);
        applyTranslations(lang);
      });
    },
  };
}());
