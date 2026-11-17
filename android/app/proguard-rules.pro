# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in /Users/bineetkeshari/Library/Android/sdk/tools/proguard/proguard-android.txt
# You can edit the include path and order by changing the proguardFiles
# directive in build.gradle.

# Retrofit & OkHttp
-dontwarn okhttp3.**
-dontwarn retrofit2.**
-keep class retrofit2.** { *; }
-keepattributes Signature
-keepattributes *Annotation*

# Gson
-keepattributes EnclosingMethod
-keepclassmembers class * {
    @com.google.gson.annotations.SerializedName <fields>;
}

# Keep Data Transfer Objects (DTOs)
-keep class ai.indoor.app.data.models.** { *; }
-keep class ai.indoor.app.domain.models.** { *; }

