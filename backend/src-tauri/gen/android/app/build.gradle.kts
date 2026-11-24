import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("rust")
}

val tauriProperties = Properties().apply {
    val propFile = file("tauri.properties")
    if (propFile.exists()) {
        propFile.inputStream().use { load(it) }
    }
}

android {
    compileSdk = 36
    namespace = "com.arounder.hifishifter"
    defaultConfig {
        manifestPlaceholders["usesCleartextTraffic"] = "false"
        applicationId = "com.arounder.hifishifter"
        // [Why 26 and not the tauri default 24] cpal 0.18's Android backend is AAudio,
        // and the NDK only ships libaaudio.so under <sysroot>/usr/lib/<abi>/<api>/ for
        // api >= 26. With minSdk 24 the link fails with "unable to find library -laaudio";
        // forcing the linker to the api-26 stub while keeping minSdk 24 would instead
        // produce a library that cannot be loaded on Android 7.x. AAudio is the only
        // backend cpal has on this platform, so 26 is the real floor.
        // Keep in sync with bundle.android.minSdkVersion in tauri.conf.json.
        minSdk = 26
        targetSdk = 36
        versionCode = tauriProperties.getProperty("tauri.android.versionCode", "1").toInt()
        versionName = tauriProperties.getProperty("tauri.android.versionName", "1.0")
    }
    buildTypes {
        getByName("debug") {
            manifestPlaceholders["usesCleartextTraffic"] = "true"
            isDebuggable = true
            isJniDebuggable = true
            isMinifyEnabled = false
            packaging {                jniLibs.keepDebugSymbols.add("*/arm64-v8a/*.so")
                jniLibs.keepDebugSymbols.add("*/armeabi-v7a/*.so")
                jniLibs.keepDebugSymbols.add("*/x86/*.so")
                jniLibs.keepDebugSymbols.add("*/x86_64/*.so")
            }
        }
        getByName("release") {
            isMinifyEnabled = true
            proguardFiles(
                *fileTree(".") { include("**/*.pro") }
                    .plus(getDefaultProguardFile("proguard-android-optimize.txt"))
                    .toList().toTypedArray()
            )
        }
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    buildFeatures {
        buildConfig = true
    }
}

rust {
    rootDirRel = "../../../"
}

dependencies {
    implementation("androidx.webkit:webkit:1.14.0")
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("androidx.activity:activity-ktx:1.10.1")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.lifecycle:lifecycle-process:2.10.0")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.4")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.0")
}

// ── Never let the packager reuse a previous APK ──────────────────────────────
//
// [Why] AGP's packager (zipflinger) **updates an existing archive in place** and
// never truncates it. When a native library changes it writes the new entry and
// leaves the old bytes behind as a hole, so the file grows by the size of the
// library on every rebuild. Measured on this project: a 463 MB debug APK became
// 917 MB after a single rebuild that only replaced libbackend_lib.so, and the
// 901 MB one before it carried ~437 MB of zeros between entries. The archive
// stays *valid* (the central directory is rewritten, so the app still installs),
// which is what makes it dangerous: nothing fails, the download just doubles.
//
// [Why in taskGraph.whenReady and not in the task's own doFirst] The packager
// keeps an inventory of what the archive already contains in
// build/intermediates/incremental/package<Variant>/tmp. Deleting the archive from
// inside doFirst leaves that inventory intact, so the packager writes only what it
// believes is missing -- measured: the APK ended up holding the native library and
// nothing else, while still reporting BUILD SUCCESSFUL. Deleting before the graph
// executes makes the packaging task genuinely out of date, which is the same
// situation as a clean build (verified twice: 930 entries, correct size).
//
// [Why the whole outputs directory] The packaging task declares directories, not
// files, and only the archives inside them go stale -- dex, resources.arsc and the
// merged native libraries all live under build/intermediates and are untouched.
// Re-packaging costs ~7 s, which is what packaging already cost.
gradle.taskGraph.whenReady {
    val packagesArchive = allTasks.any { task ->
        (task.name.startsWith("package") || task.name.startsWith("bundle")) &&
            !task.name.endsWith("Resources")
    }
    if (packagesArchive) {
        listOf("apk", "bundle").forEach { dir ->
            file("build/outputs/" + dir).deleteRecursively()
        }
    }
}

apply(from = "tauri.build.gradle.kts")