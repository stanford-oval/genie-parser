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

#include "posix.h"

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
}

JNIEXPORT void JNICALL Java_edu_stanford_nlp_sempre_PosixHelper_setuid
  (JNIEnv * env, jclass klass, jstring chuid)
{
	AutoReleaseString chuid_jstr(env, chuid);
	// null terminate chuid_jstr
	string chuid_str(chuid_jstr.data(), chuid_jstr.size());

	struct passwd *user = getpwnam(chuid_str.c_str());
	if (user == nullptr) {
		env->ThrowNew(env->FindClass("java/lang/Exception"), "Invalid user name");
		return;
	}

	if (setgid(user->pw_gid) == -1) {
		env->ThrowNew(env->FindClass("java/lang/SecurityException"), "Failed to change group ID");
		return;
	}
	if (setuid(user->pw_uid) == -1) {
		env->ThrowNew(env->FindClass("java/lang/SecurityException"), "Failed to change user ID");
		return;
	}
}

