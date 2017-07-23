from pprint import PrettyPrinter
import smtplib
from email.message import EmailMessage
from datetime import datetime
import pandas as pd


def mailsend(experiment, parameter, targetname, target, f1, auc):
    paramtable = pd.DataFrame({
        'F1': f1,
        'AUC': auc
    }, columns=['F1', 'AUC'], index=pd.Index(target, name=targetname))
    
    msg = EmailMessage()
    msg['Subject'] = "Parameter Tunning Completed."
    msg['From'] = "azureuser@cloudymiao.cloudapp.net"
    msg['To'] = "geniusxiaoguai@gmail.com"

    msg.set_content("{0}\nBonjour! The experiment \"{1}\" parameter tunning has completed."\
    "\nParameter setting:\n{2}\nCross-validation Tunning Result:\n{3}\n"\
    "Yours sincerely\nTeam parameter tunning server"\
    .format(datetime.today().strftime("%a, %d %b %Y %H:%M:%S"), experiment, parameter, paramtable.to_string()))

    msg.add_alternative("""\
    <html>
      <head></head>
      <body>
        {0}
        <p>Bonjour! The experiment "{1}" parameter tunning has completed.</p>
        <p>Parameter setting: {2}</p>
        <p>Cross-validation Tunning Result</p>
        {3}
        <br/>
        Yours sincerely
        Team parameter tunning server
      </body>
    </html>
    """.format(datetime.today().strftime("%a, %d %b %Y %H:%M:%S"), experiment, parameter, paramtable.to_html()),\
    subtype="html")

    # in case mail sending failed
    with open('experiment_completed_{0}.msg'.format(datetime.today().strftime("%Y-%m-%d_%H_%M_%S")), 'wb') as f:
        f.write(bytes(msg))

    # Send the message via our own SMTP server.
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()


def mailsendfail(experiment, parameter, exception):

    msg = EmailMessage()
    msg.set_content("""
    {0}
    Bonjour! The experiment "{1}" parameter tunning has failed. 

    Parameter:
    {2}

    Exception message:
    {3}

    Yours sincerely
    Team parameter tunning server

    """.format(datetime.today().strftime("%a, %d %b %Y %H:%M:%S"), experiment, parameter, str(exception)))

    msg['Subject'] = "Parameter Tunning Failed."
    msg['From'] = "azureuser@cloudymiao.cloudapp.net"
    msg['To'] = "geniusxiaoguai@gmail.com"

    # in case mail sending failed
    with open('experiment_failed_{0}.msg'.format(datetime.today().strftime("%Y-%m-%d_%H_%M_%S")), 'wb') as f:
        f.write(bytes(msg))

    # Send the message via our own SMTP server.
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()