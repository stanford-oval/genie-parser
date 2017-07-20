/*
 * posix.cpp
 *
 *  Created on: Jun 4, 2016
 *      Author: gcampagn
 */

#include <cstdlib>
#include <cstddef>
#include <string>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

#include <hunspell.hxx>
#include "hunspell-jni.h"

using namespace std;

namespace {
class AutoReleaseString
{
private:
	JNIEnv *m_env;
	jsize m_len;
	const char *m_str;
	jstring m_string;

public:
	AutoReleaseString(JNIEnv * env, jstring string) : m_env(env), m_len(0), m_str(nullptr), m_string(string) {
		m_len = env->GetStringUTFLength(string);
		m_str = env->GetStringUTFChars(string, nullptr);
	}
	~AutoReleaseString() {
		m_env->ReleaseStringUTFChars(m_string, m_str);
	}

	const char *data() { return m_str; }
	jsize size() { return m_len; }
};

jstring
string_to_jstring(JNIEnv *env, const std::string &str) {
    return env->NewStringUTF(str.c_str());
}
}

JNIEXPORT jlong JNICALL
Java_edu_stanford_nlp_sempre_corenlp_HunspellDictionary_nativeLoadLib(JNIEnv *env, jclass, jstring jaffPath, jstring jdicPath)
{
    AutoReleaseString affPath(env, jaffPath);
    AutoReleaseString dicPath(env, jdicPath);
    
    return (jlong) new (std::nothrow) Hunspell(affPath.data(), dicPath.data());
}

JNIEXPORT jboolean JNICALL
Java_edu_stanford_nlp_sempre_corenlp_HunspellDictionary_nativeSpell(JNIEnv *env, jclass, jlong jlib, jstring jword)
{
    Hunspell *lib = (Hunspell*)jlib;
    AutoReleaseString word(env, jword);

#if 0
    return (jboolean) lib->spell(std::string(word.data(), word.size()));
#else
    return (jboolean) lib->spell(word.data());
#endif
}

JNIEXPORT jobject JNICALL
Java_edu_stanford_nlp_sempre_corenlp_HunspellDictionary_nativeSuggest(JNIEnv *env, jclass, jlong jlib, jstring jword)
{
    Hunspell *lib = (Hunspell*)jlib;
    AutoReleaseString word(env, jword);
    jclass ArrayList = env->FindClass("Ljava/util/ArrayList;");
    jmethodID ArrayList_new = env->GetMethodID(ArrayList, "<init>", "()V");
    jmethodID ArrayList_add = env->GetMethodID(ArrayList, "add", "(Ljava/lang/Object;)Z");
    
    jobject result = env->NewObject(ArrayList, ArrayList_new);
#if 0
    auto suggestions = lib->suggest(std::string(word.data(), word.size()));
    for (const auto& suggestion : suggestions)
        env->CallBooleanMethod(result, ArrayList_add, string_to_jstring(env, suggestion));
#else
    char** suggestions;
    int num_suggestions = lib->suggest(&suggestions, word.data());
    for (int i = 0; i < num_suggestions; i++)
        env->CallBooleanMethod(result, ArrayList_add, env->NewStringUTF(suggestions[i]));
    lib->free_list(&suggestions, num_suggestions);
#endif

    return result;
}

JNIEXPORT void JNICALL
Java_edu_stanford_nlp_sempre_corenlp_HunspellDictionary_nativeFreeLib(JNIEnv *env, jclass, jlong jlib)
{
    Hunspell *lib = (Hunspell*)jlib;
    delete lib;
}

