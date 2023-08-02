from flask import Blueprint, render_template, request, flash

auth = Blueprint('auth', __name__)

@auth.route('/sign-in', methods=['GET', 'POST'])
def sign_in():
    return render_template("sign_in.html", user="MinJae")

@auth.route('/logout')
def logout():
    return render_template("logout.html")

@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form.get('email')
        nickname = request.form.get('nickname')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        # 유효성 검사
        if len(email) < 5 :
            flash("이메일은 5자 이상입니다.", category="error")
        elif len(nickname) < 2:
            flash("닉네임은 2자 이상입니다.", category="error")
        elif password1 != password2 :
            flash("비밀번호와 비밀번호재입력이 서로 다릅니다.", category="error")
        elif len(password1) < 3:
            flash("비밀번호가 너무 짧습니다.", category="error")
        else:
             flash("회원가입 완료.", category="success")  # Create User -> DB
    

    return render_template("sign_up.html")